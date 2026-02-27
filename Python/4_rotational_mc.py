#!/usr/bin/env python3
"""
4_rotational_mc.py

Rotational (slump) Monte Carlo per transect, triggered by hazard curve fits (Step 3).

Inputs
------
- config.json must contain:
    outputs.transects_dir
    paths.parameters.rot   (path to your rotational params JSON)
    paths.parameters.magnitude_csv (only if failure.mode = newmark_rs2009)

- hazard fits must exist in:
    <transects_dir>/hazard_fits/
  with files:
    tr_XXXXXX_pga_fit.npz
  and key:
    ln_pga_coeffs

Run
---
python 4_rotational_mc.py --keep-fs-checks --n-trials 10000 --n-jobs 30

Outputs
-------
<transects_dir>/tr_XXXX_rot_mcs.csv
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utilities.stats import (
    trunc_normal,
    trunc_lognormal,
    parse_transect_id,
    beta_from_segment
)

from utilities.failure_tools import (
    sample_Rur,
    sample_Rur_direct,
    kcrit_rotational,
    eta0_rotational_predictive,
    newmark_disp_rathje_saygili_2009_m,
    make_circle_arc_through_endpoints,
    slip_on_transect,
    eval_k_at_s,
    load_hazard_fit,
    sample_bounded_lognormal,
    sample_thickness_T,
    load_transect,
)

RHO_W = 1000.0  # kg/m^3


OUT_COLS = [
    "transect_id", "trial_id",
    "s0_m", "s1_m",
    "L_eff_m", "d0_m",
    "beta_deg", "rho_s", "su_ratio",
    "Rur",
    "FS0", "kcrit",
    "RPmin_yr", "k_RPmin", "failed_within_cap",
    "T_m", "w_m", "d_center_m", "eta0_m",
    "r_m", "dphi_rad",
]


# -----------------------------
# MC per transect
# -----------------------------
def run_rot_mc_for_transect(
    transect_csv,
    n_trials,
    seed,
    keep_fs_checks,
    out_fmt,
    params,
    hazard_dir,
    rp_grid,
    magValues=None,
    verbose=False,
):
    tid = int(parse_transect_id(transect_csv))
    s_all, z_all = load_transect(transect_csv)

    out_path = out_fmt.format(tid=tid)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if s_all is None:
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (tid, 0, 0)

    hz = load_hazard_fit(hazard_dir, tid)
    if hz is None:
        # no hazard -> still write empty output so pipeline can continue
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (tid, 0, 0)

    # params blocks (rotational.json)
    d0_cfg = params["d0"]
    L_cfg = params["L"]
    T_cfg = params["T"]
    rho_cfg = params["rho"]
    su_cfg = params["strength"]["su_ratio"]
    ru_cfg = params["ru"]
    geom_cfg = params.get("geometry", {}) or {}
    w_cfg = params.get("width", None)  # optional, but recommended for Step 7 volumes
    failure_cfg = params["failure"]

    failure_mode = str(failure_cfg["mode"]).lower().strip()

    if failure_mode not in ("kcrit_threshold", "newmark_rs2009"):
        raise RuntimeError(f"Unknown failure.mode='{failure_mode}'")

    sigma_ln_rs = float(failure_cfg.get("sigma_ln", 0.7))
    Dcrit_cfg = failure_cfg.get("Dcrit_m", {}) or {}

    # geometry controls
    arc_npts = int(geom_cfg.get("arc_npts", 200))
    sagitta_scale = float(geom_cfg.get("sagitta_scale", 1.0))
    max_attempts_factor = int(geom_cfg.get("max_attempts_factor", 30))

    # Ru controls
    eps_r = float(ru_cfg.get("eps_r", 0.35))
    r_gauss_std = float(ru_cfg.get("r_gauss_std", 0.25))
    clip_min = float(ru_cfg.get("clip_min", 0.0))
    clip_max = float(ru_cfg.get("clip_max", 0.95))

    # width (optional)
    if w_cfg is not None:
        w_min = float(w_cfg["w_min"])
        w_max = float(w_cfg["w_max"])
    else:
        w_min = w_max = np.nan

    depth_pos_all = -z_all
    rng = np.random.default_rng(int(seed + tid * 10007))

    out_rows = []
    attempts = 0
    max_attempts = int(n_trials) * int(max_attempts_factor)

    s_min = float(s_all[0])
    s_max = float(s_all[-1])

    rp_grid = np.asarray(rp_grid, dtype=float)
    rp_grid = np.unique(rp_grid)

    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        # sample priors
        d0 = trunc_lognormal(rng, d0_cfg["mean"], d0_cfg["std"], d0_cfg["min"], d0_cfg["max"])
        L = trunc_lognormal(rng, L_cfg["mean"], L_cfg["std"], L_cfg["min"], L_cfg["max"])
        rho_s = trunc_normal(rng, rho_cfg["mean"], rho_cfg["std"], rho_cfg["min"], rho_cfg["max"])
        su_ratio = trunc_normal(rng, su_cfg["mean"], su_cfg["std"], su_cfg["min"], su_cfg["max"])

        # place slide
        max_s0 = s_max - L
        if max_s0 <= s_min:
            continue
        cand = np.where(s_all <= max_s0)[0]
        if cand.size < 5:
            continue

        i0 = int(cand[np.argmin(np.abs(depth_pos_all[cand] - d0))])
        s0 = float(s_all[i0])

        i1 = int(np.searchsorted(s_all, s0 + L, side="left"))
        if i1 <= i0 + 2 or i1 >= len(s_all):
            continue

        s1 = float(s_all[i1])
        L_eff = float(s1 - s0)
        if L_eff <= 0.0:
            continue

        s_seg = s_all[i0:i1 + 1]
        zb_seg_real = z_all[i0:i1 + 1]  # real bathy (negative)

        # linearized seabed in slide region (Grilli-style)
        z0 = float(zb_seg_real[0])
        z1 = float(zb_seg_real[-1])
        zb_seg = z0 + (s_seg - s0) * (z1 - z0) / max((s1 - s0), 1e-9)

        beta_deg = beta_from_segment(s_seg, -zb_seg)
        if not np.isfinite(beta_deg) or beta_deg <= 0.0:
            continue

        ru_model = ru_cfg["model"]

        if ru_model == "grilli":
            Rur, rr, Rgauss = sample_Rur(beta_deg, rng, eps_r=eps_r, r_gauss_std=r_gauss_std)

        elif ru_model == "direct":
            Rur = sample_Rur_direct(
                rng,
                Rur_max=ru_cfg.get("Rur_max", 0.8),
                a=ru_cfg.get("a", 2.0),
                b=ru_cfg.get("b", 2.0),
            )
        else:
            raise ValueError(f"Unknown ru.model='{ru_model}'")

        if not np.isfinite(Rur):
            continue
        Rur = float(np.clip(Rur, clip_min, clip_max))

        # thickness proxy + slip arc
        T = sample_thickness_T(rng, T_cfg, L_eff)
        sagitta = float(max(1e-6, sagitta_scale * T))

        arc = make_circle_arc_through_endpoints(s0, z0, s1, z1, sagitta=sagitta, npts=arc_npts)
        if arc is None:
            continue

        s_arc, z_arc, r_circle, dphi, chord_len = arc
        if (not np.isfinite(r_circle)) or (r_circle <= 0.0):
            continue

        dphi_use = float(dphi)
        if np.isfinite(dphi_use) and (dphi_use > 2.0 * np.pi + 1e-6):
            dphi_use = float(np.deg2rad(dphi_use))
        if (not np.isfinite(dphi_use)) or (dphi_use <= 0.0):
            continue

        zs_seg = slip_on_transect(s_seg, s_arc, z_arc)

        # thickness relative to LINEAR seabed
        th = zb_seg - zs_seg
        if np.any(~np.isfinite(th)) or np.any(th <= 0.0):
            continue

        # stability
        FS0, kcrit = kcrit_rotational(s_seg,zb_seg,zs_seg,rho_s,su_ratio,Rur,r_circle)

        if not np.isfinite(FS0) or not np.isfinite(kcrit):
            continue
        if keep_fs_checks and (FS0 <= 1.0):
            continue

        # center + depth for eta0 uses REAL bathy
        s_center = 0.5 * (s0 + s1)
        zb_center_real = float(np.interp(s_center, s_seg, zb_seg_real))
        d_center = float(max(1.0, -zb_center_real))

        # optional width for later volume maps
        if np.isfinite(w_min) and np.isfinite(w_max):
            w = float(rng.uniform(w_min, w_max)) * L_eff
        else:
            w = np.nan

        # eta0 proxy
        eta0 = eta0_rotational_predictive(
            beta_deg=beta_deg,
            T=T,
            l=L_eff,
            d=d_center,
            r=r_circle,
            dphi=dphi_use,
            rho_s=rho_s,
            rng=rng
        )
        if not np.isfinite(eta0):
            continue

        # hazard ladder at slide center
        k_series = eval_k_at_s(hz, s_center, rp_grid)

        RPmin = np.nan
        k_RPmin = np.nan
        failed_within_cap = 0

        M_evt = np.nan
        Dcrit_m = np.nan
        D_RPmin = np.nan
        newmark_eps_ln = np.nan

        if failure_mode == "newmark_rs2009":
            if magValues is None or len(magValues) == 0:
                raise RuntimeError("failure.mode=newmark_rs2009 requires magnitude.csv to be filled")
            M_evt = float(rng.choice(magValues))

            Dcrit_m = sample_bounded_lognormal(
                rng,
                Dcrit_cfg["median"], Dcrit_cfg["sigma_ln"],
                Dcrit_cfg["min"], Dcrit_cfg["max"]
            )
            newmark_eps_ln = float(rng.normal(0.0, max(0.0, sigma_ln_rs)))

            D_series = newmark_disp_rathje_saygili_2009_m(k_series, kcrit, M_evt, newmark_eps_ln)
            D_series = np.maximum.accumulate(D_series)

            hit = np.where(D_series >= Dcrit_m)[0]
            if hit.size > 0:
                jmin = int(hit[0])
                RPmin = float(rp_grid[jmin])
                k_RPmin = float(k_series[jmin])
                D_RPmin = float(D_series[jmin])
                failed_within_cap = 1

        else:
            hit = np.where(k_series >= kcrit)[0]
            if hit.size > 0:
                jmin = int(hit[0])
                RPmin = float(rp_grid[jmin])
                k_RPmin = float(k_series[jmin])
                failed_within_cap = 1

        out_rows.append({
            "transect_id": int(tid),
            "trial_id": int(len(out_rows)),

            "s0_m": round(s0, 0),
            "s1_m": round(s1, 0),
            "L_eff_m": round(L_eff, 1),
            "d0_m": round(d0, 1),

            "beta_deg": round(beta_deg, 2),
            "rho_s": round(rho_s, 1),
            "su_ratio": round(su_ratio, 4),

            "Rur": round(Rur, 3),

            "FS0": round(FS0, 2),
            "kcrit": round(kcrit, 6),

            "RPmin_yr": int(RPmin) if np.isfinite(RPmin) else np.nan,
            "k_RPmin": round(k_RPmin, 6) if np.isfinite(k_RPmin) else np.nan,
            "failed_within_cap": int(failed_within_cap),

            "T_m": round(T, 1),
            "w_m": round(w, 1) if np.isfinite(w) else np.nan,
            "d_center_m": round(d_center, 1),
            "eta0_m": round(eta0, 3),

            "r_m": round(float(r_circle), 1),
            "dphi_rad": round(float(dphi_use), 6),

        })

    out = pd.DataFrame(out_rows, columns=OUT_COLS)
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[OK] tr_{tid:04d}: {len(out_rows)} accepted / {attempts} attempts -> {out_path}")

    return (tid, int(len(out_rows)), int(attempts))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Rotational Monte Carlo per transect (hazard-triggered).")
    ap.add_argument("--config", default="config.json", help="Path to case config.json")
    ap.add_argument("--n-trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-fs-checks", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--backend", type=str, default="loky", choices=["loky", "multiprocessing", "threading"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Load config, transects and parameters
    # Read case config
    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    transects_dir = cfg["outputs"]["transects_dir"]

    hazard_dir = transects_dir + "/hazard_fits"
    if not os.path.isdir(hazard_dir):
        raise SystemExit(f"[FAIL] Missing hazard_dir: {hazard_dir}")

    params_path = cfg["paths"]["parameters"]["rot"]
    with open(params_path, "r") as f:
        params = json.load(f)

    # --- Failure mode ---
    failure_cfg = params["failure"]
    failure_mode = failure_cfg["mode"]

    if failure_mode == "newmark_rs2009":
        mag_path = cfg["paths"]["parameters"]["magnitude_csv"]
        if not os.path.exists(mag_path):
                raise SystemExit(f"[FAIL] Missing magnitude CSV: {mag_path}")
        df_M = pd.read_csv(mag_path)
        magValues = pd.to_numeric(df_M["M_evt"], errors="coerce").to_numpy(dtype=float)
    else:
        magValues = None

    hz_cfg = params["hazard"]
    rp_min = float(hz_cfg["rp_min"])
    rp_max = float(hz_cfg["rp_max"])
    rp_step = float(hz_cfg["rp_step"])

    if not (np.isfinite(rp_min) and np.isfinite(rp_max) and np.isfinite(rp_step) and rp_step > 0 and rp_max >= rp_min):
        raise SystemExit(f"[FAIL] Bad RP ramp: rp_min={rp_min} rp_max={rp_max} rp_step={rp_step}")

    rp_grid = np.arange(rp_min, rp_max + 0.5 * rp_step, rp_step, dtype=float)

    out_fmt = os.path.join(transects_dir, "tr_{tid:04d}_rot_mcs.csv")

    files = sorted(glob.glob(os.path.join(transects_dir, "tr_*.csv")))
    files = [
        f for f in files
        if os.path.basename(f) != "index.csv"
        and "_mcs" not in os.path.basename(f)
        and "transects_over" not in os.path.basename(f)
    ]
    if not files:
        raise SystemExit(f"[FAIL] No transect CSV files found in: {transects_dir}")

    print(f"[INFO] rotational MC: {len(files)} transects")
    print(f"[INFO] n_trials={args.n_trials}  n_jobs={args.n_jobs}  backend={args.backend}")
    print("[INFO] params:", params_path)
    print("[INFO] hazard_dir:", hazard_dir)
    print(f"[INFO] RP ramp: min={rp_min} max={rp_max} step={rp_step}")
    print("[INFO] keep_fs_checks:", bool(args.keep_fs_checks))
    print("[INFO] failure.mode:", failure_mode)
    print("[INFO] outputs ->", out_fmt.format(tid=0).replace("0000", "????"))

    if args.n_jobs == 1:
        tot = 0
        for f in files:
            _, nacc, _ = run_rot_mc_for_transect(
                f, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
                params, hazard_dir, rp_grid, magValues, verbose=True
            )
            tot += nacc
        print("[DONE] total accepted trials written:", tot)
        return

    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(run_rot_mc_for_transect)(
            f, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
            params, hazard_dir, rp_grid, magValues, args.verbose
        )
        for f in files
    )
    tot = int(sum(nacc for (_, nacc, _) in results))
    print("[DONE] total accepted trials written:", tot)


if __name__ == "__main__":
    main()