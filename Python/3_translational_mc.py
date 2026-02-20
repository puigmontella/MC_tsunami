#!/usr/bin/env python3
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# -----------------------------
# Project root on PYTHONPATH
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utilities.stats import (
    trunc_normal,
    trunc_lognormal,
    parse_transect_id,
    find_col,
    beta_from_segment,
)

from utilities.failure_tools import (
    sample_Rut,
    kcrit_translational,
    eta0_translational_eq18,
)

# -----------------------------
# Defaults (fallback if params missing)
# -----------------------------
DEFAULTS = dict(
    d0=dict(mean=1247.0, std=485.0, min=50.0, max=4000.0),
    L=dict(mean=5749.0, std=3024.0, min=200.0, max=30000.0),
    T=dict(mode="ratio", tl_min=0.01, tl_max=0.08, min_m=1.0, max_m=500.0),
    phi=dict(mean=32.0, std=4.0, min=20.0, max=45.0),
    rho=dict(mean=1816.6, std=39.9, min=1400.0, max=2200.0),
    ru=dict(eps_t=0.5, r_gauss_std=0.25),
    width=dict(w_min=0.20, w_max=0.30),
    geometry=dict(max_attempts_factor=40),
)

# -----------------------------
# Small helpers
# -----------------------------
def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _resolve_path(p):
    """Resolve relative paths against PROJECT_ROOT."""
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(PROJECT_ROOT, p))

def _get(d, keys, default=None):
    """Safe nested get: keys can be ['a','b','c']."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _merge_defaults(user_params, defaults):
    """Shallow-ish merge for our known structure."""
    out = dict(defaults)
    for k, v in (user_params or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            tmp = dict(out[k])
            tmp.update(v)
            out[k] = tmp
        else:
            out[k] = v
    return out

def _load_transect(path):
    dfT = pd.read_csv(path)

    s_col = find_col(dfT, ["s_m", "dist_m", "distance_m", "s"])
    z_col = find_col(dfT, ["depth", "depth_m", "z", "bathy_m"])

    if s_col is None or z_col is None:
        raise RuntimeError(
            "{} missing s/depth columns. Found: {}".format(os.path.basename(path), list(dfT.columns))
        )

    s_all = dfT[s_col].to_numpy(float)
    z_all = dfT[z_col].to_numpy(float)  # expected negative offshore

    m = np.isfinite(s_all) & np.isfinite(z_all)
    s_all = s_all[m]
    z_all = z_all[m]
    if len(s_all) < 5:
        return None, None

    order = np.argsort(s_all)
    s_all = s_all[order]
    z_all = z_all[order]
    return s_all, z_all

def _sample_thickness_T(rng, T_cfg, L_eff):
    """
    Supported:
      T.mode = 'ratio' with tl_min/tl_max (T = tl * L_eff)
      T.mode = 'uniform' with min_m/max_m (T ~ U[min_m, max_m])
      T.mode = 'fixed' with value_m
    """
    mode = str(T_cfg.get("mode", "ratio")).lower()

    if mode == "ratio":
        tl_min = float(T_cfg.get("tl_min", 0.01))
        tl_max = float(T_cfg.get("tl_max", 0.08))
        tl = float(rng.uniform(tl_min, tl_max))
        T = tl * float(L_eff)
    elif mode == "uniform":
        mn = float(T_cfg.get("min_m", 1.0))
        mx = float(T_cfg.get("max_m", 500.0))
        T = float(rng.uniform(mn, mx))
    elif mode == "fixed":
        T = float(T_cfg.get("value_m", 50.0))
    else:
        # fallback
        tl = float(rng.uniform(0.01, 0.08))
        T = tl * float(L_eff)

    # optional clipping
    if "min_m" in T_cfg:
        T = max(T, float(T_cfg["min_m"]))
    if "max_m" in T_cfg:
        T = min(T, float(T_cfg["max_m"]))

    return float(T)

# -----------------------------
# Core MC
# -----------------------------
def run_trans_mc_for_transect(
    transect_csv,
    n_trials,
    seed,
    keep_fs_checks,
    out_fmt,
    params,
    verbose=False,
):
    tid = parse_transect_id(transect_csv)

    s_all, z_all = _load_transect(transect_csv)
    if s_all is None:
        return (tid, 0, 0)

    depth_pos_all = -z_all  # positive depth for placement
    rng = np.random.default_rng(int(seed + tid * 10007))

    out_rows = []
    attempts = 0
    max_attempts = int(n_trials * float(params["geometry"].get("max_attempts_factor", 40)))

    s_min = float(s_all[0])
    s_max = float(s_all[-1])

    d0_cfg = params["d0"]
    L_cfg  = params["L"]
    T_cfg  = params["T"]
    phi_cfg = params["phi"]
    rho_cfg = params["rho"]
    ru_cfg  = params["ru"]
    w_cfg   = params["width"]

    eps_t = float(ru_cfg.get("eps_t", 0.5))
    r_gauss_std = float(ru_cfg.get("r_gauss_std", 0.25))
    w_min = float(w_cfg.get("w_min", 0.20))
    w_max = float(w_cfg.get("w_max", 0.30))

    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        # ---- priors
        d0 = trunc_lognormal(rng, float(d0_cfg["mean"]), float(d0_cfg["std"]), float(d0_cfg["min"]), float(d0_cfg["max"]))
        L  = trunc_lognormal(rng, float(L_cfg["mean"]),  float(L_cfg["std"]),  float(L_cfg["min"]),  float(L_cfg["max"]))

        phi_deg = trunc_normal(rng, float(phi_cfg["mean"]), float(phi_cfg["std"]), float(phi_cfg["min"]), float(phi_cfg["max"]))
        rho_s   = trunc_normal(rng, float(rho_cfg["mean"]), float(rho_cfg["std"]), float(rho_cfg["min"]), float(rho_cfg["max"]))

        # ---- place slide: must fit s0 + L <= s_max
        max_s0 = s_max - L
        if max_s0 <= s_min:
            continue

        cand = np.where(s_all <= max_s0)[0]
        if cand.size < 5:
            continue

        i0 = int(cand[np.argmin(np.abs(depth_pos_all[cand] - d0))])
        s0 = float(s_all[i0])

        s1_target = s0 + L
        i1 = int(np.searchsorted(s_all, s1_target, side="left"))
        if i1 <= i0 + 2 or i1 >= len(s_all):
            continue

        s1 = float(s_all[i1])
        L_eff = float(s1 - s0)
        if L_eff <= 0.0:
            continue

        s_seg = s_all[i0:i1+1]
        zb_seg = z_all[i0:i1+1]  # seabed z (negative)
        depth_pos_seg = -zb_seg

        beta_deg = beta_from_segment(s_seg, depth_pos_seg)
        if not np.isfinite(beta_deg) or beta_deg <= 0.0:
            continue

        # ---- thickness uses L_eff (more consistent than sampled L)
        T = _sample_thickness_T(rng, T_cfg, L_eff)

        # ---- Ru sampling
        Rut, rVt, rt, alpha_t, Rgauss = sample_Rut(
            beta_deg=beta_deg,
            phi_deg=phi_deg,
            rng=rng,
            eps_t=eps_t,
            r_gauss_std=r_gauss_std
        )

        # ---- FS checks (optional)
        tb = np.tan(np.radians(beta_deg))
        tp = np.tan(np.radians(phi_deg))
        if tb <= 0.0 or not np.isfinite(tb) or not np.isfinite(tp):
            continue

        FS1 = tp / tb
        FS2 = (1.0 - Rut) * FS1

        if keep_fs_checks and ((FS1 <= 1.0) or (FS2 <= 1.0)):
            continue

        # ---- kcrit
        kcrit, FS0 = kcrit_translational(beta_deg, phi_deg, rho_s, Rut)
        if not np.isfinite(kcrit) or not np.isfinite(FS0):
            continue

        # ---- eta0
        s_center = 0.5 * (s0 + s1)
        zb_center = float(np.interp(s_center, s_seg, zb_seg))
        d_center = float(max(1.0, -zb_center))

        w_coeff = float(rng.uniform(w_min, w_max))
        w = w_coeff * L_eff

        eta0 = eta0_translational_eq18(
            beta_deg=beta_deg,
            T=T,
            l=L_eff,
            d=d_center,
            rho_s=rho_s,
            w=w
        )

        out_rows.append({
            "transect_id": tid,
            "trial_id": len(out_rows),

            "s0_m": s0,
            "s1_m": s1,

            # include both names for convenience downstream
            "L_m": L_eff,
            "L_eff_m": L_eff,

            "d0_m": d0,
            "beta_deg": float(beta_deg),
            "phi_deg": float(phi_deg),
            "rho_s": float(rho_s),

            "Rut": float(Rut),
            "rVt": float(rVt),
            "rt": float(rt),
            "alpha_t": float(alpha_t),
            "R_gauss": float(Rgauss),

            "FS1": float(FS1),
            "FS2": float(FS2),
            "FS0": float(FS0),

            "kcrit": float(kcrit),

            "T_m": float(T),
            "w_m": float(w),
            "d_center_m": float(d_center),
            "eta0_m": float(eta0),

            "failure_type": "trans",
        })

    out = pd.DataFrame(out_rows)
    out_path = out_fmt.format(tid=tid)
    out.to_csv(out_path, index=False)

    if verbose:
        print("[OK] tr_{:04d}: {} accepted / {} attempts -> {}".format(tid, len(out), attempts, out_path))

    return (tid, int(len(out)), int(attempts))

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Translational (slide) Monte Carlo per transect (kcrit + eta0), driven by config + params JSON.")
    ap.add_argument("--config", required=True, help="Path to cases/config.json")

    ap.add_argument("--transects-dir", default=None, help="Override transects directory (default from config outputs.transects_dir)")
    ap.add_argument("--params", default=None, help="Override translational params JSON (default from config paths.parameters.trans)")
    ap.add_argument("--pattern", default=None, help="Glob for transect CSVs (default: <transects_dir>/tr_*.csv)")
    ap.add_argument("--out-fmt", default=None, help="Output format (must include {tid:04d}). Default: <transects_dir>/tr_{tid:04d}_grilli_mcs.csv")

    ap.add_argument("--n-trials", type=int, default=5000, help="Accepted trials per transect")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-fs-checks", action="store_true", help="Keep only FS1>1 and FS2>1")

    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--backend", type=str, default="loky", choices=["loky", "multiprocessing", "threading"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config)
    cfg = _load_json(cfg_path)

    transects_dir = args.transects_dir
    if transects_dir is None:
        transects_dir = _get(cfg, ["outputs", "transects_dir"], "transects")
    transects_dir = _resolve_path(transects_dir)
    os.makedirs(transects_dir, exist_ok=True)

    params_path = args.params
    if params_path is None:
        params_path = _get(cfg, ["paths", "parameters", "trans"], None)
    if params_path is None:
        raise SystemExit("Config missing paths.parameters.trans (and you didn't pass --params).")
    params_path = _resolve_path(params_path)

    user_params = _load_json(params_path)
    params = _merge_defaults(user_params, DEFAULTS)

    out_fmt = args.out_fmt
    if out_fmt is None:
        out_fmt = os.path.join(transects_dir, "tr_{tid:04d}_grilli_mcs.csv")

    pattern = args.pattern if args.pattern else os.path.join(transects_dir, "tr_*.csv")
    files = sorted(glob.glob(pattern))

    # filter out non-transect files (index, maps, MC outputs)
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
        raise SystemExit("No transect CSV files found for pattern: {}".format(pattern))

    print("[INFO] translational MC: {} transects, n_trials={}, n_jobs={}, backend={}".format(
        len(files), args.n_trials, args.n_jobs, args.backend
    ))
    print("[INFO] using params:", params_path)
    print("[INFO] outputs ->", out_fmt)

    if args.n_jobs == 1:
        tot = 0
        for f in files:
            tid, nacc, att = run_trans_mc_for_transect(
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

    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(run_trans_mc_for_transect)(
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
