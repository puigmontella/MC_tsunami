#!/usr/bin/env python3
"""
4_translational_mc.py

Translational Monte Carlo per transect, triggered by a hazard curve fit.

Key fix vs your version
-----------------------
- Slide placement is now done by CENTER depth:
      d0  ~=  d_center
  This matches the meaning of d0 in your AREAS translational MC and helps make
  eta0_m distributions much more consistent between transects and areas.

Hazard fits
-----------
This script accepts either:
- tr_XXXXXX_pga_fit.npz with key "ln_pga_coeffs"
- tr_XXXXXX_pha_fit.npz with key "lnpha_coeffs"

Run
---
python 4_translational_mc.py --keep-fs-checks --n-trials 10000 --n-jobs 30
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
    beta_from_segment,
)

from utilities.failure_tools import (
    sample_Rut,
    sample_Rut_direct,
    kcrit_translational,
    eta0_translational_predictive,
    newmark_disp_rathje_saygili_2009_m,
    eval_k_at_s,
    sample_bounded_lognormal,
    sample_thickness_T,
    load_transect,
)

RHO_W = 1000.0  # kg/m^3

# Placement defaults (similar spirit to areas)
DEPTH_MATCH_TOL_M_DEFAULT = 200.0
MIN_CANDIDATES_DEFAULT = 8
FALLBACK_TOP_K_DEFAULT = 30

OUT_COLS = [
    "transect_id", "trial_id",
    "s0_m", "s1_m",
    "L_eff_m", "d0_m",
    "beta_deg", "phi_deg", "rho_s",
    "Rut",
    "FS2", "kcrit",
    "RPmin_yr", "k_RPmin", "failed_within_cap",
    "T_m", "w_m", "d_center_m", "eta0_m",
]


# -----------------------------
# Hazard loader (flexible)
# -----------------------------
def load_hazard_fit_flexible(hazard_dir, tid):
    """
    Returns a dict compatible with eval_k_at_s():
      { "path", "s_m", "ln_coeffs", "deg" }
    Supports:
      - tr_XXXXXX_pga_fit.npz with ln_pga_coeffs
      - tr_XXXXXX_pha_fit.npz with lnpha_coeffs
    """
    tid = int(tid)

    candidates = [
        (os.path.join(hazard_dir, f"tr_{tid:06d}_pga_fit.npz"),
         ["ln_pga_coeffs"]),
        (os.path.join(hazard_dir, f"tr_{tid:06d}_pha_fit.npz"),
         ["lnpha_coeffs", "ln_pha_coeffs"]),
    ]

    fpath = None
    coeff_key = None
    for p, keys in candidates:
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            for k in keys:
                if k in z.files:
                    fpath = p
                    coeff_key = k
                    break
            if fpath is not None:
                break

    if fpath is None:
        return None

    D = np.load(fpath, allow_pickle=False)

    if "s_m" not in D.files:
        raise KeyError(f"[FAIL] hazard NPZ missing 's_m': {fpath} keys={D.files}")

    C = np.asarray(D[coeff_key], dtype=float)
    if C.ndim != 2:
        raise ValueError(f"[FAIL] hazard coeffs not 2D in {fpath}: shape={C.shape}")

    deg = int(D["deg"]) if "deg" in D.files else (C.shape[0] - 1)

    return {
        "path": fpath,
        "s_m": np.asarray(D["s_m"], dtype=float),
        "ln_coeffs": C,      # (deg+1, n_pts)
        "deg": deg,
    }


# -----------------------------
# Monte Carlo run
# -----------------------------
def run_trans_mc_for_transect(
    transect_csv,
    n_trials,
    seed,
    keep_fs_checks,
    out_fmt,
    params,
    hazard_dir,
    rp_grid,
    rp_max,          # kept for compatibility (not used)
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

    hz = load_hazard_fit_flexible(hazard_dir, tid)
    if hz is None:
        # no hazard fit -> write empty so pipeline can continue
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (tid, 0, 0)

    # --- Params ---
    d0_cfg = params["d0"]
    L_cfg = params["L"]
    T_cfg = params["T"]
    phi_cfg = params["phi"]
    rho_cfg = params["rho"]
    ru_cfg = params["ru"]
    w_cfg = params["width"]
    failure_cfg = params["failure"]

    # OPTIONAL placement block (if you want to tune later)
    pcfg = params.get("placement", {}) or {}
    depth_match_tol_m = float(pcfg.get("depth_match_tol_m", DEPTH_MATCH_TOL_M_DEFAULT))
    min_candidates = int(pcfg.get("min_candidates", MIN_CANDIDATES_DEFAULT))
    fallback_top_k = int(pcfg.get("fallback_top_k", FALLBACK_TOP_K_DEFAULT))

    depth_pos_all = -z_all
    s_min = float(s_all[0])
    s_max = float(s_all[-1])

    # --- Failure mode ---
    failure_mode = str(failure_cfg["mode"]).lower().strip()
    if failure_mode not in ("kcrit_threshold", "newmark_rs2009"):
        raise RuntimeError(f"Unknown failure.mode='{failure_mode}'")

    sigma_ln_rs = float(failure_cfg.get("sigma_ln", 0.7))
    Dcrit_cfg = failure_cfg.get("Dcrit_m", {}) or {}

    # Ru and width controls
    w_min = float(w_cfg["w_min"])
    w_max = float(w_cfg["w_max"])

    # --- Clean RP grid ---
    rp_grid = np.asarray(rp_grid, dtype=float)
    rp_grid = np.unique(rp_grid)
    if rp_grid.size == 0:
        raise RuntimeError("RP grid is empty.")

    rng = np.random.default_rng(int(seed + tid * 10007))

    out_rows = []
    attempts = 0
    max_attempts = int(max(10 * n_trials, n_trials))  # allow rejects

    # usable centers (finite depth)
    m_usable = np.isfinite(s_all) & np.isfinite(depth_pos_all) & (depth_pos_all > 0.0)
    usable_idx = np.where(m_usable)[0]
    if usable_idx.size < 10:
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (tid, 0, 0)

    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        # --- Sample slide parameters ---
        d0 = trunc_lognormal(rng, d0_cfg["mean"], d0_cfg["std"], d0_cfg["min"], d0_cfg["max"])
        L = trunc_lognormal(rng, L_cfg["mean"], L_cfg["std"], L_cfg["min"], L_cfg["max"])
        phi_deg = trunc_normal(rng, phi_cfg["mean"], phi_cfg["std"], phi_cfg["min"], phi_cfg["max"])
        rho_s = trunc_normal(rng, rho_cfg["mean"], rho_cfg["std"], rho_cfg["min"], rho_cfg["max"])

        L = float(L)
        if not np.isfinite(L) or L <= 0.0:
            continue

        halfL = 0.5 * L

        # --- CENTER placement constraint ---
        candC = usable_idx[(s_all[usable_idx] >= (s_min + halfL)) & (s_all[usable_idx] <= (s_max - halfL))]
        if candC.size < 5:
            continue

        d_err = np.abs(depth_pos_all[candC] - float(d0))
        tol = min(depth_match_tol_m, 0.35 * float(d0))  # same heuristic as areas
        cand_local = candC[d_err <= tol]

        if cand_local.size >= min_candidates:
            iC = int(rng.choice(cand_local))
        else:
            k = min(fallback_top_k, candC.size)
            if k < 1:
                continue
            top = candC[np.argpartition(d_err, k - 1)[:k]]
            iC = int(rng.choice(top))

        s_center = float(s_all[iC])
        d_center = float(depth_pos_all[iC])  # by construction ~ d0

        # --- Convert center -> endpoints ---
        s0_target = s_center - halfL
        s1_target = s_center + halfL

        i0 = int(np.searchsorted(s_all, s0_target, side="left"))
        i1 = int(np.searchsorted(s_all, s1_target, side="left"))

        if i1 <= i0 + 2 or i1 >= len(s_all) or i0 < 0:
            continue

        s0 = float(s_all[i0])
        s1 = float(s_all[i1])
        L_eff = float(s1 - s0)
        if not np.isfinite(L_eff) or L_eff <= 0.0:
            continue

        # segment arrays
        s_seg = s_all[i0:i1 + 1]
        zb_seg = z_all[i0:i1 + 1]          # negative depth
        depth_pos_seg = -zb_seg            # positive depth

        # --- Local slope beta (same as before: along-transect) ---
        beta_deg = beta_from_segment(s_seg, depth_pos_seg)
        if not np.isfinite(beta_deg) or beta_deg <= 0.0:
            continue

        # --- Thickness ---
        T = sample_thickness_T(rng, T_cfg, L_eff)

        # --- Ru sampling ---
        ru_model = str(ru_cfg["model"]).lower().strip()
        if ru_model == "grilli":
            Rut, *_ = sample_Rut(
                beta_deg=beta_deg, phi_deg=phi_deg, rng=rng,
                eps_t=float(ru_cfg["eps_t"]), r_gauss_std=float(ru_cfg["r_gauss_std"]),
            )
        elif ru_model == "direct":
            Rut = sample_Rut_direct(
                rng,
                Rut_max=float(ru_cfg.get("Rut_max", 0.8)),
                a=float(ru_cfg.get("a", 2.0)),
                b=float(ru_cfg.get("b", 2.0)),
            )
        else:
            raise ValueError(f"Unknown ru.model='{ru_model}'")

        # --- Quick FS checks ---
        tb = np.tan(np.radians(beta_deg))
        tp = np.tan(np.radians(phi_deg))
        if tb <= 0.0 or (not np.isfinite(tb)) or (not np.isfinite(tp)):
            continue

        FS1 = tp / tb
        FS2 = (1.0 - float(Rut)) * FS1
        if keep_fs_checks and ((FS1 <= 1.0) or (FS2 <= 1.0)):
            continue

        # --- kcrit ---
        kcrit = kcrit_translational(beta_deg, phi_deg, rho_s, Rut)
        if not np.isfinite(kcrit):
            continue

        # --- Width ---
        w = float(rng.uniform(w_min, w_max)) * L_eff
        if not np.isfinite(w) or w <= 0.0:
            continue

        # --- Tsunami proxy eta0 ---
        eta0 = eta0_translational_predictive(beta_deg, T, L_eff, d_center, rho_s, w, RHO_W)
        if not np.isfinite(eta0):
            continue

        # --- Hazard k(RP) at slide center ---
        k_series = eval_k_at_s(hz, s_center, rp_grid)

        RPmin = np.nan
        k_RPmin = np.nan
        failed_within_cap = 0

        if failure_mode == "newmark_rs2009":
            if magValues is None or len(magValues) == 0:
                raise RuntimeError("failure.mode=newmark_rs2009 requires magnitude.csv to be filled")

            M_evt = float(rng.choice(magValues))

            Dcrit_m = sample_bounded_lognormal(
                rng,
                float(Dcrit_cfg["median"]), float(Dcrit_cfg["sigma_ln"]),
                float(Dcrit_cfg["min"]), float(Dcrit_cfg["max"])
            )
            eps_ln = float(rng.normal(0.0, max(0.0, sigma_ln_rs)))

            D_series = newmark_disp_rathje_saygili_2009_m(k_series, kcrit, M_evt, eps_ln)
            D_series = np.maximum.accumulate(D_series)

            hit = np.where(D_series >= Dcrit_m)[0]
            if hit.size > 0:
                jmin = int(hit[0])
                RPmin = float(rp_grid[jmin])
                k_RPmin = float(k_series[jmin])
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
            "d0_m": round(float(d0), 1),

            "beta_deg": round(float(beta_deg), 2),
            "phi_deg": round(float(phi_deg), 2),
            "rho_s": round(float(rho_s), 1),

            "Rut": round(float(Rut), 3),

            "FS2": round(float(FS2), 2),
            "kcrit": round(float(kcrit), 6),

            "RPmin_yr": int(RPmin) if np.isfinite(RPmin) else np.nan,
            "k_RPmin": round(float(k_RPmin), 6) if np.isfinite(k_RPmin) else np.nan,
            "failed_within_cap": int(failed_within_cap),

            "T_m": round(float(T), 1),
            "w_m": round(float(w), 1),
            "d_center_m": round(float(d_center), 1),
            "eta0_m": round(float(eta0), 3),
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
    ap = argparse.ArgumentParser(description="Translational Monte Carlo per transect (hazard-triggered).")
    ap.add_argument("--config", default="config.json", help="Path to case config.json")
    ap.add_argument("--n-trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-fs-checks", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--backend", type=str, default="loky", choices=["loky", "multiprocessing", "threading"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    transects_dir = cfg["outputs"]["transects_dir"]

    hazard_dir = os.path.join(transects_dir, "hazard_fits")
    if not os.path.isdir(hazard_dir):
        raise SystemExit(f"[FAIL] Missing hazard_dir: {hazard_dir}")

    params_path = cfg["paths"]["parameters"]["trans"]
    with open(params_path, "r") as f:
        params = json.load(f)

    # Failure mode (needs magnitudes only for Newmark)
    failure_mode = str(params["failure"]["mode"]).lower().strip()
    if failure_mode == "newmark_rs2009":
        mag_path = cfg["paths"]["parameters"]["magnitude_csv"]
        if not os.path.exists(mag_path):
            raise SystemExit(f"[FAIL] Missing magnitude CSV: {mag_path}")
        df_M = pd.read_csv(mag_path)
        magValues = pd.to_numeric(df_M["M_evt"], errors="coerce").to_numpy(dtype=float)
        magValues = magValues[np.isfinite(magValues)]
    else:
        magValues = None

    hz_cfg = params["hazard"]
    rp_min = float(hz_cfg["rp_min"])
    rp_max = float(hz_cfg["rp_max"])
    rp_step = float(hz_cfg["rp_step"])

    if not (np.isfinite(rp_min) and np.isfinite(rp_max) and np.isfinite(rp_step) and rp_step > 0 and rp_max >= rp_min):
        raise SystemExit(f"[FAIL] Bad RP ramp: rp_min={rp_min} rp_max={rp_max} rp_step={rp_step}")

    rp_grid = np.arange(rp_min, rp_max + 0.5 * rp_step, rp_step, dtype=float)

    out_fmt = os.path.join(transects_dir, "tr_{tid:04d}_trans_mcs.csv")

    files = sorted(glob.glob(os.path.join(transects_dir, "tr_*.csv")))
    files = [
        f for f in files
        if os.path.basename(f) != "index.csv"
        and "_mcs" not in os.path.basename(f)
        and "transects_over" not in os.path.basename(f)
    ]
    if not files:
        raise SystemExit(f"[FAIL] No transect CSV files found in: {transects_dir}")

    print(f"[INFO] translational MC: {len(files)} transects")
    print(f"[INFO] n_trials={args.n_trials}  n_jobs={args.n_jobs}  backend={args.backend}")
    print("[INFO] params:", params_path)
    print("[INFO] hazard_dir:", hazard_dir)
    print(f"[INFO] RP ramp: min={rp_min} max={rp_max} step={rp_step}")
    print("[INFO] keep_fs_checks:", bool(args.keep_fs_checks))
    print("[INFO] failure.mode:", failure_mode)

    if args.n_jobs == 1:
        tot = 0
        for fcsv in files:
            _, nacc, _ = run_trans_mc_for_transect(
                fcsv, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
                params, hazard_dir, rp_grid, rp_max, magValues, verbose=True
            )
            tot += nacc
        print("[DONE] total accepted trials written:", tot)
        return

    prefer = "threads" if args.backend == "threading" else "processes"
    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, prefer=prefer, verbose=10)(
        delayed(run_trans_mc_for_transect)(
            fcsv, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
            params, hazard_dir, rp_grid, rp_max, magValues, args.verbose
        )
        for fcsv in files
    )
    tot = int(sum(nacc for (_, nacc, _) in results))
    print("[DONE] total accepted trials written:", tot)


if __name__ == "__main__":
    main()