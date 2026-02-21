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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

RHO_W = 1000.0  # consistent with utilities.failure_tools


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

    # Failure model:
    #  - kcrit_threshold: failure when k(RP) >= kcrit
    #  - newmark_rs2009 : failure when D(RP) >= Dcrit, D from Rathje & Saygili (2009) PGA+M regression
    failure=dict(
        mode="kcrit_threshold",  # "kcrit_threshold" | "newmark_rs2009"
        M=dict(dist="normal", mean=7.0, std=0.5, min=6.0, max=8.5),
        Dcrit_m=dict(dist="lognormal", median=0.8, sigma_ln=0.8, min=0.05, max=20.0),
        sigma_ln=0.7,
    ),
)

OUT_COLS = [
    "transect_id", "trial_id",
    "s0_m", "s1_m",
    "L_m", "L_eff_m",
    "d0_m", "beta_deg", "phi_deg", "rho_s",
    "Rut", "rVt", "rt", "alpha_t", "R_gauss",
    "FS1", "FS2", "FS0",
    "kcrit",
    "RPmin_yr", "k_RPmin", "failed_within_cap",
    "RPprev_yr", "k_RPprev",
    "FS3", "FS4", "RPmin_check_ok",
    "failure_mode", "M_evt", "Dcrit_m",
    "D_RPmin_m", "D_RPprev_m", "newmark_eps_ln",
    "T_m", "w_m", "d_center_m", "eta0_m",
    "failure_type",
]


# -----------------------------
# Small helpers
# -----------------------------
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_path(p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(PROJECT_ROOT, p))

def _get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _deep_merge(base, upd):
    out = dict(base) if isinstance(base, dict) else {}
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _merge_defaults(user_params, defaults):
    return _deep_merge(defaults, user_params or {})

def _load_transect(path):
    dfT = pd.read_csv(path)
    s_col = find_col(dfT, ["s_m", "dist_m", "distance_m", "s"])
    z_col = find_col(dfT, ["depth", "depth_m", "z", "bathy_m"])
    if s_col is None or z_col is None:
        raise RuntimeError(
            f"{os.path.basename(path)} missing s/depth columns. Found: {list(dfT.columns)}"
        )
    s_all = dfT[s_col].to_numpy(float)
    z_all = dfT[z_col].to_numpy(float)  # negative offshore
    m = np.isfinite(s_all) & np.isfinite(z_all)
    s_all = s_all[m]
    z_all = z_all[m]
    if len(s_all) < 5:
        return None, None
    order = np.argsort(s_all)
    return s_all[order], z_all[order]

def _sample_thickness_T(rng, T_cfg, L_eff):
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
        tl = float(rng.uniform(0.01, 0.08))
        T = tl * float(L_eff)

    if "min_m" in T_cfg:
        T = max(T, float(T_cfg["min_m"]))
    if "max_m" in T_cfg:
        T = min(T, float(T_cfg["max_m"]))
    return float(T)

def _sample_bounded_lognormal(rng, median, sigma_ln, vmin, vmax, max_tries=80):
    med = float(median)
    sig = float(sigma_ln)
    lo = float(vmin)
    hi = float(vmax)
    if not (np.isfinite(med) and med > 0 and np.isfinite(sig) and sig >= 0):
        return np.nan
    mu = np.log(med)
    for _ in range(int(max_tries)):
        x = float(rng.lognormal(mean=mu, sigma=sig))
        if np.isfinite(x) and (x >= lo) and (x <= hi):
            return x
    x = float(rng.lognormal(mean=mu, sigma=sig))
    if not np.isfinite(x):
        return np.nan
    return float(np.clip(x, lo, hi))


# -----------------------------
# Hazard fit (PHA curve) helpers
# -----------------------------
def load_hazard_fit(hazard_dir, tid):
    f = os.path.join(hazard_dir, f"tr_{int(tid):06d}_pha_fit.npz")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing hazard fit for transect {tid}: {f}")
    D = np.load(f)
    return {
        "s_m": D["s_m"].astype(float),
        "lnpha_coeffs": D["lnpha_coeffs"].astype(float),  # (deg+1, n_pts)
        "deg": int(D["deg"]),
    }

def eval_k_at_s(hz, s_center_m, rp_years):
    sgrid = hz["s_m"]
    C = hz["lnpha_coeffs"]  # (deg+1, n_pts)

    s_center_m = float(np.clip(float(s_center_m), float(sgrid[0]), float(sgrid[-1])))

    c_at_s = np.array([np.interp(s_center_m, sgrid, C[i, :]) for i in range(C.shape[0])], dtype=float)

    rp = np.asarray(rp_years, dtype=float)
    prob = 1.0 / rp
    lnpha = np.polyval(c_at_s, prob)
    k = np.exp(lnpha)
    k = np.maximum.accumulate(k)  # robust monotonic
    return k

def FS_translational(beta_deg, phi_deg, rho_s, Rut, k):
    beta = np.radians(float(beta_deg))
    phi = np.radians(float(phi_deg))
    Ru = float(Rut)
    s = float(rho_s) / RHO_W

    tb = np.tan(beta)
    tp = np.tan(phi)
    k = float(k)

    den = ((s - 1.0) * tb + k * s)
    if not np.isfinite(den) or den == 0.0:
        return np.nan

    num = ((s - 1.0) * (1.0 - Ru) - k * s * tb)
    FS = (num / den) * tp
    return float(FS) if np.isfinite(FS) else np.nan


# -----------------------------
# Newmark displacement model (PGA-only + Magnitude)
# Rathje & Saygili (2009) regression (as implemented in OpenQuake).
# PGA_g is in g, same as hazard fit output k
# critaccel_g is ky in g, here ky = kcrit
# output in meters
# -----------------------------
def newmark_disp_rathje_saygili_2009_m(PGA_g, critaccel_g, M, eps_ln=0.0):
    PGA_g = np.asarray(PGA_g, dtype=float)
    crit = float(critaccel_g)
    M = float(M)

    D = np.zeros_like(PGA_g, dtype=float)
    m = np.isfinite(PGA_g) & (PGA_g > 0.0) & (PGA_g > crit) & np.isfinite(crit) & (crit > 0.0)

    if not np.any(m):
        return D

    r = crit / PGA_g[m]  # ky/PGA
    lnD_cm = (
        4.89
        - 4.85 * r
        - 19.64 * r**2
        + 42.49 * r**3
        - 29.06 * r**4
        + 0.72 * np.log(PGA_g[m])
        + 0.89 * (M - 6.0)
        + float(eps_ln)
    )
    D[m] = np.exp(lnD_cm) / 100.0  # cm -> m
    return D


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
    hazard_dir,
    rp_grid,
    rp_check_delta,
    rp_cap,
    verbose=False,
):
    tid = parse_transect_id(transect_csv)

    s_all, z_all = _load_transect(transect_csv)
    out_path = out_fmt.format(tid=tid)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if s_all is None:
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (tid, 0, 0)

    hz = load_hazard_fit(hazard_dir, tid)

    depth_pos_all = -z_all  # positive depth
    rng = np.random.default_rng(int(seed + tid * 10007))

    out_rows = []
    attempts = 0
    max_attempts = int(n_trials * float(params["geometry"].get("max_attempts_factor", 40)))

    s_min = float(s_all[0])
    s_max = float(s_all[-1])

    d0_cfg = params["d0"]
    L_cfg = params["L"]
    T_cfg = params["T"]
    phi_cfg = params["phi"]
    rho_cfg = params["rho"]
    ru_cfg = params["ru"]
    w_cfg = params["width"]

    # failure config
    failure_cfg = params.get("failure", {}) or {}
    failure_mode = str(failure_cfg.get("mode", "kcrit_threshold")).lower().strip()
    if failure_mode in ("newmark", "displacement"):
        failure_mode = "newmark_rs2009"

    if failure_mode not in ("kcrit_threshold", "newmark_rs2009"):
        raise RuntimeError(
            f"Unknown failure.mode='{failure_mode}'. Use 'kcrit_threshold' or 'newmark_rs2009'."
        )

    Mcfg = failure_cfg.get("M", {}) or {}
    Dcrit_cfg = failure_cfg.get("Dcrit_m", {}) or {}
    sigma_ln_D = float(failure_cfg.get("sigma_ln", 0.7))

    eps_t = float(ru_cfg.get("eps_t", 0.5))
    r_gauss_std = float(ru_cfg.get("r_gauss_std", 0.25))
    w_min = float(w_cfg.get("w_min", 0.20))
    w_max = float(w_cfg.get("w_max", 0.30))

    rp_grid = np.asarray(rp_grid, dtype=float)
    rp_grid = rp_grid[np.isfinite(rp_grid) & (rp_grid > 0.0)]
    rp_grid = np.unique(rp_grid)

    if np.isfinite(rp_cap) and rp_cap > 0:
        rp_grid = rp_grid[rp_grid <= float(rp_cap)]
    if rp_grid.size == 0:
        raise RuntimeError("RP grid is empty after filtering/capping. Check rp_min/rp_max/rp_step/rp_cap.")

    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        # priors
        d0 = trunc_lognormal(
            rng, float(d0_cfg["mean"]), float(d0_cfg["std"]), float(d0_cfg["min"]), float(d0_cfg["max"])
        )
        L = trunc_lognormal(
            rng, float(L_cfg["mean"]), float(L_cfg["std"]), float(L_cfg["min"]), float(L_cfg["max"])
        )
        phi_deg = trunc_normal(
            rng, float(phi_cfg["mean"]), float(phi_cfg["std"]), float(phi_cfg["min"]), float(phi_cfg["max"])
        )
        rho_s = trunc_normal(
            rng, float(rho_cfg["mean"]), float(rho_cfg["std"]), float(rho_cfg["min"]), float(rho_cfg["max"])
        )

        # place slide: s0 + L <= s_max
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

        s_seg = s_all[i0:i1 + 1]
        zb_seg = z_all[i0:i1 + 1]  # negative
        depth_pos_seg = -zb_seg

        beta_deg = beta_from_segment(s_seg, depth_pos_seg)
        if not np.isfinite(beta_deg) or beta_deg <= 0.0:
            continue

        # thickness
        T = _sample_thickness_T(rng, T_cfg, L_eff)

        # Ru sampling
        Rut, rVt, rt, alpha_t, Rgauss = sample_Rut(
            beta_deg=beta_deg, phi_deg=phi_deg, rng=rng, eps_t=eps_t, r_gauss_std=r_gauss_std
        )

        # FS checks (Grilli stage 1–2)
        tb = np.tan(np.radians(beta_deg))
        tp = np.tan(np.radians(phi_deg))
        if tb <= 0.0 or not np.isfinite(tb) or not np.isfinite(tp):
            continue

        FS1 = tp / tb
        FS2 = (1.0 - Rut) * FS1
        if keep_fs_checks and ((FS1 <= 1.0) or (FS2 <= 1.0)):
            continue

        # kcrit
        kcrit, FS0 = kcrit_translational(beta_deg, phi_deg, rho_s, Rut)
        if not np.isfinite(kcrit) or not np.isfinite(FS0):
            continue

        # center and depth
        s_center = 0.5 * (s0 + s1)
        zb_center = float(np.interp(s_center, s_seg, zb_seg))
        d_center = float(max(1.0, -zb_center))

        # width
        w_coeff = float(rng.uniform(w_min, w_max))
        w = w_coeff * L_eff

        # eta0
        eta0 = eta0_translational_eq18(
            beta_deg=beta_deg, T=T, l=L_eff, d=d_center, rho_s=rho_s, w=w
        )
        if not np.isfinite(eta0):
            continue

        # hazard k(RP)
        k_series = eval_k_at_s(hz, s_center, rp_grid)

        # outputs init
        FS3 = np.nan
        FS4 = np.nan
        RPprev = np.nan
        k_RPprev = np.nan

        M_evt = np.nan
        Dcrit_m = np.nan
        D_RPmin = np.nan
        D_RPprev = np.nan
        newmark_eps_ln = np.nan

        # Trigger model
        if not np.isfinite(kcrit) or kcrit <= 0.0:
            RPmin = 0.0
            k_RPmin = 0.0
            failed_within_cap = 1
            RPmin_check_ok = np.nan
            FS3 = FS_translational(beta_deg, phi_deg, rho_s, Rut, 0.0)
            FS4 = FS3

        else:
            if failure_mode == "newmark_rs2009":
                # magnitude
                if str(Mcfg.get("dist", "normal")).lower() != "normal":
                    raise RuntimeError("failure.M currently supports only dist='normal'.")

                M_evt = trunc_normal(
                    rng,
                    float(Mcfg.get("mean", 7.0)),
                    float(Mcfg.get("std", 0.5)),
                    float(Mcfg.get("min", 6.0)),
                    float(Mcfg.get("max", 8.5)),
                )

                # capacity
                if str(Dcrit_cfg.get("dist", "lognormal")).lower() != "lognormal":
                    raise RuntimeError("failure.Dcrit_m currently supports only dist='lognormal'.")

                Dcrit_m = _sample_bounded_lognormal(
                    rng,
                    median=float(Dcrit_cfg.get("median", 0.8)),
                    sigma_ln=float(Dcrit_cfg.get("sigma_ln", 0.8)),
                    vmin=float(Dcrit_cfg.get("min", 0.05)),
                    vmax=float(Dcrit_cfg.get("max", 20.0)),
                )

                # residual (one per trial)
                newmark_eps_ln = float(rng.normal(loc=0.0, scale=max(0.0, sigma_ln_D)))

                D_series = newmark_disp_rathje_saygili_2009_m(
                    PGA_g=k_series, critaccel_g=float(kcrit), M=float(M_evt), eps_ln=newmark_eps_ln
                )
                D_series = np.maximum.accumulate(D_series)

                hit = np.where(D_series >= Dcrit_m)[0]
                if hit.size == 0:
                    RPmin = np.nan
                    k_RPmin = np.nan
                    failed_within_cap = 0
                    RPmin_check_ok = np.nan
                else:
                    jmin = int(hit[0])
                    RPmin = float(rp_grid[jmin])
                    k_RPmin = float(k_series[jmin])
                    D_RPmin = float(D_series[jmin])
                    failed_within_cap = 1

                    RPprev = float(max(1.0, RPmin - float(rp_check_delta)))
                    k_RPprev = float(eval_k_at_s(hz, s_center, np.array([RPprev], float))[0])

                    D_RPprev = float(newmark_disp_rathje_saygili_2009_m(
                        PGA_g=np.array([k_RPprev], float),
                        critaccel_g=float(kcrit),
                        M=float(M_evt),
                        eps_ln=newmark_eps_ln
                    )[0])

                    FS3 = FS_translational(beta_deg, phi_deg, rho_s, Rut, k_RPmin)
                    FS4 = FS_translational(beta_deg, phi_deg, rho_s, Rut, k_RPprev)

                    RPmin_check_ok = 1 if (np.isfinite(D_RPprev) and (D_RPprev < Dcrit_m)) else 0

            else:
                # kcrit threshold mode
                hit = np.where(k_series >= kcrit)[0]
                if hit.size == 0:
                    RPmin = np.nan
                    k_RPmin = np.nan
                    failed_within_cap = 0
                    RPmin_check_ok = np.nan
                else:
                    jmin = int(hit[0])
                    RPmin = float(rp_grid[jmin])
                    k_RPmin = float(k_series[jmin])
                    failed_within_cap = 1

                    RPprev = float(max(1.0, RPmin - float(rp_check_delta)))
                    k_RPprev = float(eval_k_at_s(hz, s_center, np.array([RPprev], float))[0])

                    FS3 = FS_translational(beta_deg, phi_deg, rho_s, Rut, k_RPmin)
                    FS4 = FS_translational(beta_deg, phi_deg, rho_s, Rut, k_RPprev)

                    RPmin_check_ok = 1 if (np.isfinite(FS4) and FS4 > 1.0) else 0

        out_rows.append({
            "transect_id": int(tid),
            "trial_id": int(len(out_rows)),

            "s0_m": float(s0),
            "s1_m": float(s1),

            "L_m": float(L_eff),
            "L_eff_m": float(L_eff),

            "d0_m": float(d0),
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

            "RPmin_yr": float(RPmin) if np.isfinite(RPmin) else np.nan,
            "k_RPmin": float(k_RPmin) if np.isfinite(k_RPmin) else np.nan,
            "failed_within_cap": int(failed_within_cap),

            "RPprev_yr": float(RPprev) if np.isfinite(RPprev) else np.nan,
            "k_RPprev": float(k_RPprev) if np.isfinite(k_RPprev) else np.nan,

            "FS3": float(FS3) if np.isfinite(FS3) else np.nan,
            "FS4": float(FS4) if np.isfinite(FS4) else np.nan,
            "RPmin_check_ok": float(RPmin_check_ok) if np.isfinite(RPmin_check_ok) else np.nan,

            "failure_mode": str(failure_mode),
            "M_evt": float(M_evt) if np.isfinite(M_evt) else np.nan,
            "Dcrit_m": float(Dcrit_m) if np.isfinite(Dcrit_m) else np.nan,
            "D_RPmin_m": float(D_RPmin) if np.isfinite(D_RPmin) else np.nan,
            "D_RPprev_m": float(D_RPprev) if np.isfinite(D_RPprev) else np.nan,
            "newmark_eps_ln": float(newmark_eps_ln) if np.isfinite(newmark_eps_ln) else np.nan,

            "T_m": float(T),
            "w_m": float(w),
            "d_center_m": float(d_center),
            "eta0_m": float(eta0),

            "failure_type": "trans",
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
    ap = argparse.ArgumentParser(
        description="Translational Monte Carlo per transect (Grilli) with hazard triggering via RP ladder.\n"
                    "Failure modes:\n"
                    "  - kcrit_threshold: fail when k(RP) >= kcrit (classic)\n"
                    "  - newmark_rs2009 : fail when D(RP) >= Dcrit, with D from Rathje & Saygili (2009) PGA+M\n"
    )
    ap.add_argument("--config", required=True, help="Path to cases/config.json")

    ap.add_argument("--transects-dir", default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--out-fmt", default=None)

    ap.add_argument("--hazard-dir", default=None)

    ap.add_argument("--n-trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-fs-checks", action="store_true")

    ap.add_argument("--rp-min", type=float, default=None)
    ap.add_argument("--rp-max", type=float, default=None)
    ap.add_argument("--rp-step", type=float, default=None)
    ap.add_argument("--rp-cap", type=float, default=None)
    ap.add_argument("--rp-check-delta", type=float, default=10.0)

    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--backend", type=str, default="loky",
                    choices=["loky", "multiprocessing", "threading"])
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    cfg_path = _resolve_path(args.config)
    cfg = _load_json(cfg_path)

    transects_dir = args.transects_dir
    if transects_dir is None:
        transects_dir = _get(cfg, ["outputs", "transects_dir"], "transects")
    transects_dir = _resolve_path(transects_dir)
    os.makedirs(transects_dir, exist_ok=True)

    hazard_dir = args.hazard_dir
    if hazard_dir is None:
        hazard_dir = _get(cfg, ["outputs", "hazard_dir"], os.path.join(transects_dir, "hazard_fits"))
    hazard_dir = _resolve_path(hazard_dir)
    if not os.path.isdir(hazard_dir):
        raise SystemExit(f"Missing hazard_dir: {hazard_dir}")

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

    rp_min = float(args.rp_min) if args.rp_min is not None else float(_get(cfg, ["pga_loading", "rp_min"], 50.0))
    rp_max = float(args.rp_max) if args.rp_max is not None else float(_get(cfg, ["pga_loading", "rp_max"], 750.0))
    rp_step = float(args.rp_step) if args.rp_step is not None else float(_get(cfg, ["pga_loading", "rp_step"], 10.0))
    rp_cap = float(args.rp_cap) if args.rp_cap is not None else float(_get(cfg, ["pga_loading", "rp_cap"], rp_max))

    if not (np.isfinite(rp_min) and np.isfinite(rp_max) and np.isfinite(rp_step) and rp_step > 0 and rp_max >= rp_min):
        raise SystemExit(f"Bad RP ladder params: rp_min={rp_min}, rp_max={rp_max}, rp_step={rp_step}")

    rp_grid = np.arange(rp_min, rp_max + 0.5 * rp_step, rp_step, dtype=float)

    fm = str((params.get("failure", {}) or {}).get("mode", "kcrit_threshold"))
    print(f"[INFO] translational MC: {len(files)} transects, n_trials={args.n_trials}, n_jobs={args.n_jobs}, backend={args.backend}")
    print("[INFO] using params:", params_path)
    print("[INFO] hazard_dir:", hazard_dir)
    print(f"[INFO] RP ladder: min={rp_min} max={rp_max} step={rp_step} cap={rp_cap} (years), check_delta={args.rp_check_delta}yr")
    print("[INFO] keep_fs_checks:", bool(args.keep_fs_checks))
    print("[INFO] failure.mode:", fm)
    print("[INFO] outputs ->", out_fmt)

    if args.n_jobs == 1:
        tot = 0
        for f in files:
            _, nacc, _ = run_trans_mc_for_transect(
                f,
                n_trials=args.n_trials,
                seed=args.seed,
                keep_fs_checks=args.keep_fs_checks,
                out_fmt=out_fmt,
                params=params,
                hazard_dir=hazard_dir,
                rp_grid=rp_grid,
                rp_check_delta=args.rp_check_delta,
                rp_cap=rp_cap,
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
            hazard_dir=hazard_dir,
            rp_grid=rp_grid,
            rp_check_delta=args.rp_check_delta,
            rp_cap=rp_cap,
            verbose=args.verbose
        )
        for f in files
    )

    tot = int(sum(nacc for (_, nacc, _) in results))
    print("[DONE] total accepted trials written:", tot)


if __name__ == "__main__":
    main()