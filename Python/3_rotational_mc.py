#!/usr/bin/env python3
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

# project root on PYTHONPATH (so imports work when running from anywhere)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utilities.config import load_config, abspath

from utilities.stats import (
    trunc_normal,
    trunc_lognormal,
    parse_transect_id,
    find_col,
    beta_from_segment,
)

from utilities.failure_tools import (
    sample_Rur,
    make_circle_arc_through_endpoints,
    slip_on_transect,
    rotational_fs0_kcrit_eq10,
    eta0_rotational_eq19,
)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _load_transect(path):
    dfT = pd.read_csv(path)

    s_col = find_col(dfT, ["s_m", "dist_m", "distance_m", "s"])
    z_col = find_col(dfT, ["depth", "depth_m", "z", "bathy_m"])

    if s_col is None or z_col is None:
        raise RuntimeError(
            "{} missing s/depth columns. Found: {}".format(os.path.basename(path), list(dfT.columns))
        )

    s_all = dfT[s_col].to_numpy(float)
    z_all = dfT[z_col].to_numpy(float)  # often negative offshore

    m = np.isfinite(s_all) & np.isfinite(z_all)
    s_all = s_all[m]
    z_all = z_all[m]
    if len(s_all) < 5:
        return None, None

    order = np.argsort(s_all)
    return s_all[order], z_all[order]


def run_rotational_mc_for_transect(
    transect_csv,
    n_trials,
    seed,
    keep_fs_checks,
    out_fmt,
    params,
    verbose=False
):
    tid = parse_transect_id(transect_csv)

    s_all, z_all = _load_transect(transect_csv)
    if s_all is None:
        return (tid, 0, 0)

    depth_pos_all = -z_all  # positive depth
    rng = np.random.default_rng(int(seed + tid * 10007))

    # ---- params unpack
    d0p = params["d0"]
    Lp  = params["L"]
    Tp  = params["T"]
    rhop = params["rho"]
    sup = params["strength"]["su_ratio"]
    rup = params["ru"]
    geom = params.get("geometry", {})
    eta0p = params.get("eta0", {})

    arc_npts = int(geom.get("arc_npts", 200))
    sagitta_scale = float(geom.get("sagitta_scale", 1.0))
    max_attempts_factor = int(geom.get("max_attempts_factor", 30))

    tl_min = float(Tp.get("tl_min", 0.05))
    tl_max = float(Tp.get("tl_max", 0.25))
    T_min_m = float(Tp.get("min_m", 0.2))
    T_max_m = float(Tp.get("max_m", 1.0e9))

    eps_r = float(rup.get("eps_r", 0.35))
    r_gauss_std = float(rup.get("r_gauss_std", 0.25))
    ru_clip_min = float(rup.get("clip_min", 0.0))
    ru_clip_max = float(rup.get("clip_max", 0.95))

    # informational / future-proof (eta0_rotational_eq19 already takes rng)
    _ = float(eta0p.get("lognormal_sigma", 0.25))

    out_rows = []
    attempts = 0
    max_attempts = int(n_trials) * int(max_attempts_factor)

    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        # -----------------------------
        # Sample priors
        # -----------------------------
        d0 = trunc_lognormal(rng, d0p["mean"], d0p["std"], d0p["min"], d0p["max"])
        L  = trunc_lognormal(rng, Lp["mean"],  Lp["std"],  Lp["min"],  Lp["max"])
        tl = float(rng.uniform(tl_min, tl_max))

        rho_s = trunc_normal(rng, rhop["mean"], rhop["std"], rhop["min"], rhop["max"])
        su_ratio = trunc_normal(rng, sup["mean"], sup["std"], sup["min"], sup["max"])

        # -----------------------------
        # Place slide along transect
        # -----------------------------
        i0 = int(np.argmin(np.abs(depth_pos_all - d0)))
        s0 = float(s_all[i0])

        s1_target = s0 + L
        if s1_target >= s_all[-1]:
            continue

        i1 = int(np.searchsorted(s_all, s1_target, side="left"))
        if i1 <= i0 + 2:
            continue

        s1 = float(s_all[i1])
        L_eff = float(s1 - s0)
        if L_eff <= 0.0:
            continue

        s_seg = s_all[i0:i1 + 1]
        zb_seg = z_all[i0:i1 + 1]       # seabed z (likely negative)
        depth_pos_seg = -zb_seg

        beta_deg = beta_from_segment(s_seg, depth_pos_seg)
        if not np.isfinite(beta_deg):
            continue

        # -----------------------------
        # Sample rotational Ru
        # -----------------------------
        Rur, rr, Rgauss = sample_Rur(beta_deg, rng, eps_r=eps_r, r_gauss_std=r_gauss_std)
        if np.isfinite(Rur):
            Rur = float(np.clip(Rur, ru_clip_min, ru_clip_max))

        # -----------------------------
        # Build slip arc
        # -----------------------------
        T_guess = max(T_min_m, tl * L_eff)
        T_guess = min(T_guess, T_max_m)
        sagitta = float(sagitta_scale * T_guess)

        arc = make_circle_arc_through_endpoints(
            s0, float(zb_seg[0]),
            s1, float(zb_seg[-1]),
            sagitta=sagitta,
            npts=arc_npts
        )
        if arc is None:
            continue

        s_arc, z_arc, r_circle, dphi, chord_len = arc
        zs_seg = slip_on_transect(s_seg, s_arc, z_arc)

        thickness = zb_seg - zs_seg
        if np.any(~np.isfinite(thickness)) or np.any(thickness <= 0.0):
            ok = False
            for fac in (1.5, 2.0, 3.0):
                sag2 = sagitta * fac
                arc2 = make_circle_arc_through_endpoints(
                    s0, float(zb_seg[0]),
                    s1, float(zb_seg[-1]),
                    sagitta=sag2,
                    npts=arc_npts
                )
                if arc2 is None:
                    continue
                s_arc2, z_arc2, r2, dphi2, chord_len2 = arc2
                zs2 = slip_on_transect(s_seg, s_arc2, z_arc2)
                th2 = zb_seg - zs2
                if np.all(np.isfinite(th2)) and np.all(th2 > 0.0):
                    s_arc, z_arc, r_circle, dphi, chord_len = s_arc2, z_arc2, r2, dphi2, chord_len2
                    zs_seg = zs2
                    thickness = th2
                    ok = True
                    break
            if not ok:
                continue

        # -----------------------------
        # Rotational stability: FS0 + kcrit
        # -----------------------------
        FS0, kcrit, N, A, B = rotational_fs0_kcrit_eq10(
            s_seg=s_seg,
            z_bathy_seg=zb_seg,
            z_slip_seg=zs_seg,
            rho_s=rho_s,
            su_ratio=su_ratio,
            Rur=Rur,
            r_circle=r_circle
        )

        if not np.isfinite(FS0) or not np.isfinite(kcrit):
            continue

        if keep_fs_checks and (FS0 <= 1.0):
            continue

        # -----------------------------
        # Rotational eta0 proxy
        # -----------------------------
        s_center = 0.5 * (s0 + s1)
        zb_center = float(np.interp(s_center, s_seg, zb_seg))
        d_center = float(max(1.0, -zb_center))

        T_max = float(np.max(thickness))
        eta0 = eta0_rotational_eq19(
            beta_deg=beta_deg,
            T=T_max,
            l=L_eff,
            d=d_center,
            r=r_circle,
            dphi=dphi,
            rho_s=rho_s,
            rng=rng
        )

        out_rows.append({
            "transect_id": tid,
            "trial_id": len(out_rows),

            "s0_m": s0,
            "s1_m": s1,
            "L_eff_m": L_eff,
            "d0_m": d0,

            "beta_deg": beta_deg,
            "rho_s": rho_s,
            "su_ratio": su_ratio,
            "Rur": Rur,
            "rr_max": rr,
            "R_gauss": Rgauss,

            "r_m": r_circle,
            "dphi_rad": dphi,
            "T_max_m": T_max,
            "d_center_m": d_center,

            "FS0": FS0,
            "kcrit": kcrit,
            "eta0_m": eta0,

            "failure_type": "rot"
        })

    out = pd.DataFrame(out_rows)
    out_path = out_fmt.format(tid=tid)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    if verbose:
        print("[OK] tr_{:04d}: {} accepted / {} attempts -> {}".format(tid, len(out), attempts, out_path))

    return (tid, int(len(out)), int(attempts))


def main():
    ap = argparse.ArgumentParser(description="Rotational (slump) Monte Carlo per transect (config-driven).")

    ap.add_argument("--config", required=True, help="Path to cases/config.json")

    # ✅ added: allow pipeline (or user) to override the params file explicitly
    ap.add_argument(
        "--params",
        default=None,
        help="Override rotational parameters JSON (else use config paths.parameters.rot)"
    )

    ap.add_argument("--transects-dir", default=None,
                    help="Override transects output dir (else from config outputs.transects_dir)")
    ap.add_argument("--pattern", default=None,
                    help="Glob for transect CSVs (default: <transects_dir>/tr_*.csv)")
    ap.add_argument("--out-fmt", default=None,
                    help="Output format (must include {tid:04d}); default in transects_dir")

    ap.add_argument("--n-trials", type=int, default=5000, help="Accepted trials per transect")
    ap.add_argument("--seed", type=int, default=1234, help="Base RNG seed")
    ap.add_argument("--keep-fs-checks", action="store_true", help="Keep only FS0>1 (stable before triggering)")

    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over transects")
    ap.add_argument("--backend", type=str, default="loky", choices=["loky", "multiprocessing", "threading"],
                    help="joblib backend for transect-level parallelism")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # ---- resolve outputs/transects dir
    if args.transects_dir is None:
        transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"])
    else:
        transects_dir = os.path.abspath(args.transects_dir)

    # ✅ changed: params path = CLI override > config
    if args.params:
        rot_param_path = os.path.abspath(args.params)
    else:
        rot_param_path = abspath(cfg, cfg["paths"]["parameters"]["rot"])

    params = _load_json(rot_param_path)

    # ---- patterns/outputs
    pattern = args.pattern if args.pattern else os.path.join(transects_dir, "tr_*.csv")
    files = sorted(glob.glob(pattern))

    # filter out non-transect files
    clean = []
    for f in files:
        b = os.path.basename(f)
        if b == "index.csv":
            continue
        if "_mcs" in b:
            continue
        if "transects_over" in b:
            continue
        clean.append(f)
    files = clean

    if not files:
        raise SystemExit(f"No transect CSV files found for pattern: {pattern}")

    out_fmt = args.out_fmt
    if out_fmt is None:
        out_fmt = os.path.join(transects_dir, "tr_{tid:04d}_grilli_rot_mcs.csv")

    print("[INFO] rotational MC: {} transects | n_trials={} | n_jobs={} | backend={}".format(
        len(files), args.n_trials, args.n_jobs, args.backend
    ))
    print("[INFO] using parameters:", rot_param_path)
    print("[INFO] outputs to:", out_fmt.format(tid=0).replace("0000", "????"))

    if args.n_jobs == 1:
        tot = 0
        for f in files:
            tid, nacc, att = run_rotational_mc_for_transect(
                f,
                n_trials=args.n_trials,
                seed=args.seed,
                keep_fs_checks=args.keep_fs_checks,
                out_fmt=out_fmt,
                params=params,
                verbose=True
            )
            tot += nacc
        print("[DONE] total accepted trials written:", tot)
        return

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(run_rotational_mc_for_transect)(
            f,
            n_trials=args.n_trials,
            seed=args.seed,
            keep_fs_checks=args.keep_fs_checks,
            out_fmt=out_fmt,
            params=params,
            verbose=args.verbose
        )
        for f in files
    )

    tot = int(sum(nacc for (_, nacc, _) in results))
    print("[DONE] total accepted trials written:", tot)


if __name__ == "__main__":
    main()
