#!/usr/bin/env python3
"""
4_translational_mc.py

Translational Monte Carlo per transect, triggered by a hazard curve fit.

Inputs
------
- config.json must contain:
    outputs.transects_dir
    paths.parameters.trans   (path to your translational params JSON)

- params JSON (paths.parameters.trans) must contain:
    d0, L, T, phi, rho, ru, width, failure
    hazard: rp_min, rp_max, rp_step, rp_max


Run:
----
python 4_translational_mc.py --keep-fs-checks --n_trials 10000 --n_jobs 30


Hazard fits
-----------
This script expects hazard fit NPZ files in:
    <transects_dir>/hazard_fits/

It will accept either naming/key style:
- tr_XXXXXX_pha_fit.npz with key "lnpha_coeffs"
- tr_XXXXXX_pga_fit.npz with key "ln_pga_coeffs"

Outputs


Help
----
python ../../Python/4_translational_mc.py -h

"""

from importlib.resources import path
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
    FS_translational,
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
    "beta_deg", "phi_deg", "rho_s",
    "Rut",
    "FS2", "kcrit",
    "RPmin_yr", "k_RPmin", "failed_within_cap",
    "T_m", "w_m", "d_center_m", "eta0_m",
]


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
    rp_max,
    magValues=None,
    verbose=False,
):
    # --- Identify transect + load profile ---
    tid = int(parse_transect_id(transect_csv))
    s_all, z_all = load_transect(transect_csv)

    out_path = out_fmt.format(tid=tid)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if s_all is None:
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (tid, 0, 0)

    # --- Hazard (PGA vs RP) ---
    hz = load_hazard_fit(hazard_dir, tid)

    # --- Params ---
    d0_cfg = params["d0"]
    L_cfg = params["L"]
    T_cfg = params["T"]
    phi_cfg = params["phi"]
    rho_cfg = params["rho"]
    ru_cfg = params["ru"]
    w_cfg = params["width"]
    failure_cfg = params["failure"]

    depth_pos_all = -z_all

    # --- random generator ---
    rng = np.random.default_rng(int(seed + tid * 10007))

    out_rows = []
    attempts = 0
    max_attempts = int(n_trials)

    s_min = float(s_all[0])
    s_max = float(s_all[-1])

    # --- Failure mode ---
    failure_mode = failure_cfg["mode"]
    if failure_mode not in ("kcrit_threshold", "newmark_rs2009"):
        raise RuntimeError(f"Unknown failure.mode='{failure_mode}'")


    # the following parameters are only used for the newmark_proxy failure mode
    sigma_ln_rs = float(failure_cfg.get("sigma_ln", 0.7))
    Dcrit_cfg = failure_cfg.get("Dcrit_m", {}) or {}




    # Ru and width controls
    eps_t = float(ru_cfg["eps_t"])
    r_gauss_std = float(ru_cfg["r_gauss_std"])
    w_min = float(w_cfg["w_min"])
    w_max = float(w_cfg["w_max"])

    # --- Clean RP grid ---
    rp_grid = np.asarray(rp_grid, dtype=float)
    rp_grid = np.unique(rp_grid)
    if rp_grid.size == 0:
        raise RuntimeError("RP grid is empty.")
    rp0 = float(rp_grid[0])

    # =========================
    # Monte Carlo accept/reject
    # =========================
    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        # --- Sample random slide parameters ---
        d0 = trunc_lognormal(rng, d0_cfg["mean"], d0_cfg["std"], d0_cfg["min"], d0_cfg["max"])
        L = trunc_lognormal(rng, L_cfg["mean"], L_cfg["std"], L_cfg["min"], L_cfg["max"])
        phi_deg = trunc_normal(rng, phi_cfg["mean"], phi_cfg["std"], phi_cfg["min"], phi_cfg["max"])
        rho_s = trunc_normal(rng, rho_cfg["mean"], rho_cfg["std"], rho_cfg["min"], rho_cfg["max"])

        # --- Choose slide location along transect ---
        max_s0 = s_max - L
        if max_s0 <= s_min:
            continue

        # candidates for s0 start anywhere from s_min to s_max-L
        cand = np.where(s_all <= max_s0)[0]
        if cand.size < 5:
            continue

        # pick s0 where depth is closest to d0
        i0 = int(cand[np.argmin(np.abs(depth_pos_all[cand] - d0))])
        s0 = float(s_all[i0])

        # end index for s0+L
        i1 = int(np.searchsorted(s_all, s0 + L, side="left"))
        if i1 <= i0 + 2 or i1 >= len(s_all):
            continue

        # effective length m (the one that is actually used)
        s1 = float(s_all[i1])
        L_eff = float(s1 - s0)
        if L_eff <= 0.0:
            continue

        # segment arrays
        s_seg = s_all[i0:i1 + 1]
        zb_seg = z_all[i0:i1 + 1]          # (negative depth)
        depth_pos_seg = -zb_seg            # positive depth

        # --- Local slope beta ---
        beta_deg = beta_from_segment(s_seg, depth_pos_seg)
        if not np.isfinite(beta_deg) or beta_deg <= 0.0:
            continue

        # --- Thickness ---
        T = sample_thickness_T(rng, T_cfg, L_eff)

        # --- Ru sampling ---
        ru_model = ru_cfg["model"]

        if ru_model == "grilli":
            Rut, rVt, rt, alpha_t, Rgauss = sample_Rut(
                beta_deg=beta_deg, phi_deg=phi_deg, rng=rng,
                eps_t=ru_cfg["eps_t"], r_gauss_std=ru_cfg["r_gauss_std"],
            )
        elif ru_model == "direct":
            Rut = sample_Rut_direct(
                rng,
                Rut_max=ru_cfg.get("Rut_max", 0.8),
                a=ru_cfg.get("a", 2.0),
                b=ru_cfg.get("b", 2.0),
            )

        else:
            raise ValueError(f"Unknown ru.model='{ru_model}'")


        # --- Quick FS checks ---
        tb = np.tan(np.radians(beta_deg))
        tp = np.tan(np.radians(phi_deg))
        if tb <= 0.0 or (not np.isfinite(tb)) or (not np.isfinite(tp)):
            continue

        FS1 = tp / tb
        FS2 = (1.0 - Rut) * FS1
        if keep_fs_checks and ((FS1 <= 1.0) or (FS2 <= 1.0)):
            continue

        # --- kcrit  ---
        kcrit = kcrit_translational(beta_deg, phi_deg, rho_s, Rut)


        # --- Center depth ---
        s_center = 0.5 * (s0 + s1)
        zb_center = np.interp(s_center, s_seg, zb_seg)
        d_center = -zb_center

        # --- Width ---
        w = float(rng.uniform(w_min, w_max)) * L_eff

        # --- Tsunami proxy eta0 ---
        eta0 = eta0_translational_predictive(beta_deg, T, L_eff, d_center, rho_s, w, RHO_W)
        if not np.isfinite(eta0):
            continue

        # --- Hazard k(RP) at slide center for each RP ---
        k_series = eval_k_at_s(hz, s_center, rp_grid)

        # init outputs
        # defaults 
        RPmin = np.nan # minimum return period (yr) at which failure occur
        k_RPmin = np.nan # hazard k at RPmin
        failed_within_cap = 0 # flag for whether failure occurs within the RP grid (1) or not (0)

        # --- Newmark-only fields (remain NaN if failure_mode = 'kcrit_threshold') ---
        M_evt = np.nan          # Earthquake magnitude sampled for this trial (used in Rathje & Saygili 2009 model)
        Td_s = np.nan           # Shaking duration (s), used only in the proxy Newmark model
        Dcrit_m = np.nan        # Critical permanent displacement capacity (m) for failure
        D_RPmin = np.nan        # Predicted Newmark displacement at RPmin (m)
        D_RPprev = np.nan       # Predicted Newmark displacement at RPprev (m)
        newmark_eps_ln = np.nan # Lognormal residual term (epsilon) added to ln(D) in the Newmark regression

        # =====================
        # Trigger / Failure step
        # =====================
        # Inputs already computed for this trial:
        #   - rp_grid   : array of return periods (years), e.g. [100,110,...,750]
        #   - k_series  : hazard shaking level at slide center for each RP (PGA in g units)
        #   - kcrit     : yield/critical acceleration (in g units) from slope+material+pore pressure
        #
        # Goal of this block:
        #   - decide if/when the slide fails as RP increases
        #   - output the first RP on the ramp that triggers failure (RPmin)
        #   - store the corresponding hazard level k_RPmin (and displacement D_RPmin for Newmark)
        #
        # two trigger models:
        #   (A) newmark_rs2009 : displacement-based Newmark using Rathje & Saygili (2009) regression
        #   (B) kcrit_threshold: simple pseudo-static threshold (PGA >= kcrit)

        if failure_mode == "newmark_rs2009":

            # --- Sample event-specific random variables for this trial ---
            # M_evt     : earthquake magnitude for this trial (taken from openQuake)
            # Dcrit_m   : displacement capacity (how much permanent displacement causes failure)
            # eps_ln    : regression residual (log-space noise) for Newmark displacement model
            if magValues is None or len(magValues) == 0:
                raise RuntimeError("failure.mode=newmark_rs2009 requires magnitude.csv to be filled")
            M_evt = float(rng.choice(magValues))

            Dcrit_m = sample_bounded_lognormal(rng, Dcrit_cfg["median"], Dcrit_cfg["sigma_ln"],
                                            Dcrit_cfg["min"], Dcrit_cfg["max"])
            newmark_eps_ln = float(rng.normal(0.0, max(0.0, sigma_ln_rs)))

            # --- Compute predicted Newmark permanent displacement D for each RP ---
            # D_series[j] = D(PGA=k_series[j], ky=kcrit, M=M_evt, eps=newmark_eps_ln)
            D_series = newmark_disp_rathje_saygili_2009_m(
                k_series, kcrit, M_evt, newmark_eps_ln
            )

            # Enforce that D does not decrease as RP increases 
            D_series = np.maximum.accumulate(D_series)

            # --- Find the first RP where displacement exceeds capacity (failure) ---
            hit = np.where(D_series >= Dcrit_m)[0]
            if hit.size > 0:
                jmin = int(hit[0])                 # first index crossing threshold
                RPmin = float(rp_grid[jmin])       # first RP that triggers failure
                k_RPmin = float(k_series[jmin])    # PGA at that RP
                D_RPmin = float(D_series[jmin])    # displacement at that RP
                failed_within_cap = 1              # failure occurred somewhere on the RP ladder


        else:
            # --- Simple pseudo-static trigger ---
            # Failure happens as soon as PGA at the slide center exceeds the yield threshold kcrit:
            #   k_series[j] >= kcrit
            hit = np.where(k_series >= kcrit)[0]
            if hit.size > 0:
                jmin = int(hit[0])
                RPmin = float(rp_grid[jmin])
                k_RPmin = float(k_series[jmin])
                failed_within_cap = 1

    # --- Save row ---
        out_rows.append({
            # IDs
            "transect_id": int(tid),
            "trial_id": int(len(out_rows)),

            # Geometry / placement
            "s0_m": round(s0,0),
            "s1_m": round(s1,0 ),
            "L_eff_m": round(L_eff,1),
            "d0_m": round(d0,1),
            "beta_deg": round(beta_deg,2),

            # Material
            "phi_deg": round(phi_deg,2),
            "rho_s": round(rho_s,1),

            # Pore pressure weakening (the only essential one)
            "Rut": round(Rut,3),

            # Stability + trigger
            "FS2": round(FS2,2),
            "kcrit": round(kcrit,6),

            "RPmin_yr": int(RPmin) if np.isfinite(RPmin) else np.nan,
            "k_RPmin": round(k_RPmin,6) if np.isfinite(k_RPmin) else np.nan,
            "failed_within_cap": int(failed_within_cap),


            # Tsunami source proxy
            "T_m": round(T,1),
            "w_m": round(w,1),
            "d_center_m": round(d_center,1),
            "eta0_m": round(eta0,3),

        })
    # --- Write output ---
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

    params_path = cfg["paths"]["parameters"]["trans"]
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

    out_fmt = transects_dir + "/tr_{tid:04d}_trans_mcs.csv"

    files = sorted(glob.glob(transects_dir + "/tr_*.csv"))

    #we only take the transect CSV files, skip index.csv and any existing MC outputs
    files = [
        f for f in files
        if os.path.basename(f) != "index.csv"
        and "_mcs" not in os.path.basename(f)
        and "transects_over" not in os.path.basename(f)
    ]
    if not files:
        raise SystemExit(f"[FAIL] No transect CSV files found in: {transects_dir}")
    

    # Print info on the translational MC run
    fm = str(params["failure"]["mode"])
    print(f"[INFO] translational MC: {len(files)} transects")
    print(f"[INFO] n_trials={args.n_trials}  n_jobs={args.n_jobs}  backend={args.backend}")
    print("[INFO] params:", params_path)
    print("[INFO] hazard_dir:", hazard_dir)
    print(f"[INFO] RP rump: min={rp_min} max={rp_max} step={rp_step}")
    print("[INFO] keep_fs_checks:", bool(args.keep_fs_checks))
    print("[INFO] failure.mode:", fm)

    if args.n_jobs == 1:
        tot = 0
        for f in files:
            _, nacc, _ = run_trans_mc_for_transect(
                f, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
                params, hazard_dir, rp_grid, rp_max, magValues, verbose=True
            )
            tot += nacc
        print("[DONE] total accepted trials written:", tot)
        return

    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(run_trans_mc_for_transect)(
            f, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
            params, hazard_dir, rp_grid, rp_max, magValues, args.verbose
        )
        for f in files
    )
    tot = int(sum(nacc for (_, nacc, _) in results))
    print("[DONE] total accepted trials written:", tot)


if __name__ == "__main__":
    main()