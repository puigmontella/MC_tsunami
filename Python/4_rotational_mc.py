#!/usr/bin/env python3
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
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

# -----------------------------
# Defaults (fallback if params missing)
# -----------------------------
DEFAULTS = dict(
    d0=dict(mean=1247.0, std=485.0, min=50.0, max=4000.0),
    L=dict(mean=5749.0, std=3024.0, min=200.0, max=30000.0),
    T=dict(mode="ratio", tl_min=0.05, tl_max=0.25, min_m=1.0, max_m=500.0),
    rho=dict(mean=1816.6, std=39.9, min=1400.0, max=2200.0),
    strength=dict(model="undrained_ratio", su_ratio=dict(mean=0.207, std=0.010, min=0.17, max=0.24)),
    ru=dict(model="gaussian_peak", eps_r=0.35, r_gauss_std=0.25, clip_min=0.0, clip_max=0.95),
    geometry=dict(arc_npts=200, sagitta_scale=1.0, max_attempts_factor=30),
    eta0=dict(model="eq19_proxy", lognormal_sigma=0.25),

    # Failure model:
    # - kcrit_threshold: fail when k(RP) >= kcrit  (classic Grilli threshold)
    # - newmark_rs2009 : fail when D(RP) >= Dcrit, with D from Rathje & Saygili (2009) PGA+M regression
    failure=dict(
        mode="kcrit_threshold",  # "kcrit_threshold" | "newmark_rs2009"
        M=dict(dist="normal", mean=7.0, std=0.5, min=6.0, max=8.5),
        Dcrit_m=dict(dist="lognormal", median=0.8, sigma_ln=0.8, min=0.05, max=20.0),
        sigma_ln=0.7,
    ),
)

OUT_COLS = [
    "transect_id","trial_id",
    "s0_m","s1_m","L_eff_m","d0_m",
    "beta_deg","rho_s","su_ratio",
    "Rur","rr_max","R_gauss",
    "T_guess_m","sagitta_m","T_max_m",
    "r_m","dphi_rad","d_center_m",
    "FS0","kcrit",

    # trigger / hazard
    "RPmin_yr","k_RPmin","failed_within_cap",
    "RPprev_yr","k_RPprev",
    "RPmin_is_lower_bound",
    "RPmin_check_ok","RPprev_is_below_rpmin",  # meaning depends on failure_mode
    "FS3","FS4",

    # eta0
    "eta0_m",

    # newmark extras (NaN if not used)
    "failure_mode","M_evt","Dcrit_m","D_RPmin_m","D_RPprev_m","newmark_eps_ln",

    "failure_type",
]

# -----------------------------
# Helpers
# -----------------------------
def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

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

def _load_transect(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    dfT = pd.read_csv(path)
    s_col = find_col(dfT, ["s_m", "dist_m", "distance_m", "s"])
    z_col = find_col(dfT, ["depth", "depth_m", "z", "bathy_m"])
    if s_col is None or z_col is None:
        raise RuntimeError(f"{os.path.basename(path)} missing s/depth columns. Found: {list(dfT.columns)}")
    s_all = dfT[s_col].to_numpy(float)
    z_all = dfT[z_col].to_numpy(float)
    m = np.isfinite(s_all) & np.isfinite(z_all)
    s_all, z_all = s_all[m], z_all[m]
    if len(s_all) < 5:
        return None, None
    o = np.argsort(s_all)
    return s_all[o], z_all[o]

def _sample_T(rng, T_cfg, L_eff: float) -> float:
    mode = str(T_cfg.get("mode", "ratio")).lower()
    if mode == "ratio":
        tl = float(rng.uniform(float(T_cfg.get("tl_min", 0.05)), float(T_cfg.get("tl_max", 0.25))))
        T = tl * float(L_eff)
    elif mode == "uniform":
        T = float(rng.uniform(float(T_cfg.get("min_m", 1.0)), float(T_cfg.get("max_m", 500.0))))
    elif mode == "fixed":
        T = float(T_cfg.get("value_m", 50.0))
    else:
        tl = float(rng.uniform(0.05, 0.25))
        T = tl * float(L_eff)
    if "min_m" in T_cfg:
        T = max(T, float(T_cfg["min_m"]))
    if "max_m" in T_cfg:
        T = min(T, float(T_cfg["max_m"]))
    return float(T)

def _sample_bounded_lognormal(rng, median, sigma_ln, vmin, vmax, max_tries=80) -> float:
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

def _load_hazard_index(hazard_dir: str):
    idx_path = os.path.join(hazard_dir, "hazard_fit_index.csv")
    if not os.path.exists(idx_path):
        return {}, idx_path
    idx = pd.read_csv(idx_path)
    mp = {}
    if "transect_id" in idx.columns and "out_npz" in idx.columns:
        for _, r in idx.iterrows():
            try:
                tid = int(r["transect_id"])
                p = str(r["out_npz"])
                # resolve relative -> hazard_dir
                if p and (not os.path.isabs(p)):
                    p = os.path.join(hazard_dir, p)
                mp[tid] = p
            except Exception:
                pass
    return mp, idx_path

def _haz_k_from_npz(npz, s_q: float, rp_year: float) -> float:
    prob = 1.0 / float(rp_year)
    sfit = npz["s_m"].astype(float)
    C = npz["lnpha_coeffs"].astype(float)  # (deg+1, n_pts)
    s_q = float(np.clip(float(s_q), float(sfit[0]), float(sfit[-1])))
    coeff = np.array([np.interp(s_q, sfit, C[i, :]) for i in range(C.shape[0])], dtype=float)
    lnpha = np.polyval(coeff, prob)
    return float(np.exp(lnpha)) if np.isfinite(lnpha) else np.nan

def _haz_k_series(npz, s_q: float, rp_ladder: list) -> np.ndarray:
    k = np.array([_haz_k_from_npz(npz, s_q, rp) for rp in rp_ladder], dtype=float)
    if k.size == 0:
        return k
    # robust monotonic: replace non-finite with previous, then enforce non-decreasing
    if not np.isfinite(k[0]):
        k[0] = 0.0
    for i in range(1, k.size):
        if not np.isfinite(k[i]):
            k[i] = k[i - 1]
        else:
            k[i] = max(k[i], k[i - 1])
    return k

# -----------------------------
# Newmark displacement model (PGA-only + Magnitude)
# Rathje & Saygili (2009) regression used by OpenQuake for Newmark displacement.
# Inputs:
#   PGA_g: PGA in units of g (same as hazard fit output k)
#   critaccel_g: yield acceleration in g (kcrit)
#   M: magnitude
# Output:
#   permanent displacement in meters
# -----------------------------
def newmark_disp_rathje_saygili_2009_m(PGA_g, critaccel_g: float, M: float, eps_ln: float = 0.0) -> np.ndarray:
    PGA_g = np.asarray(PGA_g, dtype=float)
    crit = float(critaccel_g)
    M = float(M)

    # if crit <= 0: treat as always sliding for any PGA > 0
    if not np.isfinite(crit) or crit <= 0.0:
        D = np.zeros_like(PGA_g, dtype=float)
        mpos = np.isfinite(PGA_g) & (PGA_g > 0.0)
        D[mpos] = np.inf
        return D

    D = np.zeros_like(PGA_g, dtype=float)
    m = np.isfinite(PGA_g) & (PGA_g > crit) & (PGA_g > 0.0)
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
def run_rotational_mc_for_transect(
    transect_csv: str,
    n_trials: int,
    seed: int,
    keep_fs_checks: bool,
    out_fmt: str,
    params: dict,
    hazard_npz_path: Optional[str],
    rp_ladder: list,
    rp_min: float,
    rp_cap: float,
    check_delta: float,
    verbose: bool = False,
):
    tid = int(parse_transect_id(transect_csv))

    s_all, z_all = _load_transect(transect_csv)
    out_path = out_fmt.format(tid=tid)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if s_all is None:
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (tid, 0, 0)

    npz = np.load(hazard_npz_path) if (hazard_npz_path and os.path.exists(hazard_npz_path)) else None

    depth_pos_all = -z_all
    rng = np.random.default_rng(int(seed + tid * 10007))

    d0p = params["d0"]
    Lp = params["L"]
    Tp = params["T"]
    rhop = params["rho"]
    sup = params["strength"]["su_ratio"]
    rup = params["ru"]
    geom = params.get("geometry", {})

    failure_cfg = params.get("failure", {}) or {}
    failure_mode = str(failure_cfg.get("mode", "kcrit_threshold")).lower().strip()
    if failure_mode in ("newmark", "displacement"):
        failure_mode = "newmark_rs2009"
    if failure_mode in ("newmark_proxy", "proxy"):
        raise RuntimeError("failure.mode=newmark_proxy is NOT supported in rotational. Use 'newmark_rs2009' or 'kcrit_threshold'.")

    Mcfg = failure_cfg.get("M", {}) or {}
    Dcrit_cfg = failure_cfg.get("Dcrit_m", {}) or {}
    sigma_ln_D = float(failure_cfg.get("sigma_ln", 0.7))

    arc_npts = int(geom.get("arc_npts", 200))
    sagitta_scale = float(geom.get("sagitta_scale", 1.0))
    max_attempts_factor = int(geom.get("max_attempts_factor", 30))

    eps_r = float(rup.get("eps_r", 0.35))
    r_gauss_std = float(rup.get("r_gauss_std", 0.25))
    ru_clip_min = float(rup.get("clip_min", 0.0))
    ru_clip_max = float(rup.get("clip_max", 0.95))

    out_rows = []
    attempts = 0
    max_attempts = int(n_trials) * int(max_attempts_factor)

    s_min = float(s_all[0])
    s_max = float(s_all[-1])

    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        # priors
        d0 = trunc_lognormal(rng, d0p["mean"], d0p["std"], d0p["min"], d0p["max"])
        L = trunc_lognormal(rng, Lp["mean"], Lp["std"], Lp["min"], Lp["max"])
        rho_s = trunc_normal(rng, rhop["mean"], rhop["std"], rhop["min"], rhop["max"])
        su_ratio = trunc_normal(rng, sup["mean"], sup["std"], sup["min"], sup["max"])

        # place slide
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
        if L_eff <= 0:
            continue

        s_seg = s_all[i0:i1 + 1]
        zb_seg_real = z_all[i0:i1 + 1]  # real bathy (negative)

        # ---- GRILLI BASIC: linearized seabed over the slide (plane)
        z0 = float(zb_seg_real[0])
        z1 = float(zb_seg_real[-1])
        zb_seg = z0 + (s_seg - s0) * (z1 - z0) / max((s1 - s0), 1e-9)

        beta_deg = beta_from_segment(s_seg, -zb_seg)
        if not np.isfinite(beta_deg) or beta_deg <= 0:
            continue

        # Ru
        Rur, rr, Rgauss = sample_Rur(beta_deg, rng, eps_r=eps_r, r_gauss_std=r_gauss_std)
        if not np.isfinite(Rur):
            continue
        Rur = float(np.clip(Rur, ru_clip_min, ru_clip_max))

        # T and arc (sagitta = T)
        T_guess = _sample_T(rng, Tp, L_eff)
        sagitta = float(max(1e-6, sagitta_scale * T_guess))

        arc = make_circle_arc_through_endpoints(s0, z0, s1, z1, sagitta=sagitta, npts=arc_npts)
        if arc is None:
            continue
        s_arc, z_arc, r_circle, dphi, chord_len = arc

        zs_seg = slip_on_transect(s_seg, s_arc, z_arc)

        # thickness relative to LINEAR seabed
        th = zb_seg - zs_seg
        if np.any(~np.isfinite(th)):
            continue
        h = 0.5 * (th[:-1] + th[1:])
        if np.any(h <= 0.0):
            continue
        T_max = float(np.max(th))

        # stability
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

        # hazard trigger at slide center
        s_center = 0.5 * (s0 + s1)

        # defaults
        RPmin = np.nan
        k_RPmin = np.nan
        failed_within_cap = 0

        RPprev = np.nan
        k_RPprev = np.nan

        RPmin_is_lb = 0
        RPmin_check_ok = np.nan
        below_flag = np.nan  # k_RPprev<kcrit or D_RPprev<Dcrit

        FS3 = np.nan
        FS4 = np.nan

        # newmark extras
        M_evt = np.nan
        Dcrit_m = np.nan
        D_RPmin = np.nan
        D_RPprev = np.nan
        newmark_eps_ln = np.nan

        if npz is not None:
            # compute k ladder once
            k_series = _haz_k_series(npz, s_center, rp_ladder)

            # if kcrit is weird (<=0), treat as immediate trigger at rp_min for both modes
            if (not np.isfinite(kcrit)) or (kcrit <= 0.0):
                RPmin = float(rp_min)
                # find index for rp_min in ladder (assume ladder starts at rp_min)
                j0 = 0
                k_RPmin = float(k_series[j0]) if k_series.size > 0 else np.nan
                failed_within_cap = 1
                RPmin_is_lb = 1

                RPprev = float(rp_min)
                k_RPprev = float(k_RPmin)

                if np.isfinite(N) and np.isfinite(A) and np.isfinite(B) and np.isfinite(k_RPmin) and (A + k_RPmin * B) > 0:
                    FS3 = float(N / (A + k_RPmin * B))
                    FS4 = FS3

                if failure_mode == "newmark_rs2009":
                    # still fill demand fields if desired
                    M_evt = trunc_normal(
                        rng,
                        float(Mcfg.get("mean", 7.0)),
                        float(Mcfg.get("std", 0.5)),
                        float(Mcfg.get("min", 6.0)),
                        float(Mcfg.get("max", 8.5)),
                    )
                    Dcrit_m = _sample_bounded_lognormal(
                        rng,
                        median=float(Dcrit_cfg.get("median", 0.8)),
                        sigma_ln=float(Dcrit_cfg.get("sigma_ln", 0.8)),
                        vmin=float(Dcrit_cfg.get("min", 0.05)),
                        vmax=float(Dcrit_cfg.get("max", 20.0)),
                    )
                    newmark_eps_ln = float(rng.normal(loc=0.0, scale=max(0.0, sigma_ln_D)))
                    D_RPmin = float(newmark_disp_rathje_saygili_2009_m(
                        PGA_g=np.array([k_RPmin], float),
                        critaccel_g=float(kcrit),
                        M=float(M_evt),
                        eps_ln=newmark_eps_ln
                    )[0])
                    D_RPprev = D_RPmin
                    below_flag = 0.0
                    RPmin_check_ok = np.nan

            else:
                if failure_mode == "newmark_rs2009":
                    # sample magnitude
                    if str(Mcfg.get("dist", "normal")).lower() != "normal":
                        raise RuntimeError("failure.M currently supports only dist='normal'.")

                    M_evt = trunc_normal(
                        rng,
                        float(Mcfg.get("mean", 7.0)),
                        float(Mcfg.get("std", 0.5)),
                        float(Mcfg.get("min", 6.0)),
                        float(Mcfg.get("max", 8.5)),
                    )

                    # sample Dcrit
                    if str(Dcrit_cfg.get("dist", "lognormal")).lower() != "lognormal":
                        raise RuntimeError("failure.Dcrit_m currently supports only dist='lognormal'.")

                    Dcrit_m = _sample_bounded_lognormal(
                        rng,
                        median=float(Dcrit_cfg.get("median", 0.8)),
                        sigma_ln=float(Dcrit_cfg.get("sigma_ln", 0.8)),
                        vmin=float(Dcrit_cfg.get("min", 0.05)),
                        vmax=float(Dcrit_cfg.get("max", 20.0)),
                    )

                    newmark_eps_ln = float(rng.normal(loc=0.0, scale=max(0.0, sigma_ln_D)))

                    D_series = newmark_disp_rathje_saygili_2009_m(
                        PGA_g=k_series,
                        critaccel_g=float(kcrit),
                        M=float(M_evt),
                        eps_ln=newmark_eps_ln
                    )
                    D_series = np.maximum.accumulate(D_series)

                    hit = np.where(D_series >= Dcrit_m)[0]
                    if hit.size > 0:
                        jmin = int(hit[0])
                        RPmin = float(rp_ladder[jmin])
                        k_RPmin = float(k_series[jmin])
                        D_RPmin = float(D_series[jmin])
                        failed_within_cap = 1

                        RPmin_is_lb = 1 if abs(RPmin - float(rp_min)) <= 1e-12 else 0

                        # RPprev (for check)
                        RPprev = float(max(float(rp_min), RPmin - float(check_delta)))
                        k_RPprev = float(_haz_k_from_npz(npz, s_center, RPprev))
                        if not np.isfinite(k_RPprev):
                            k_RPprev = float(k_RPmin)

                        D_RPprev = float(newmark_disp_rathje_saygili_2009_m(
                            PGA_g=np.array([k_RPprev], float),
                            critaccel_g=float(kcrit),
                            M=float(M_evt),
                            eps_ln=newmark_eps_ln
                        )[0])

                        below_flag = 1.0 if (np.isfinite(D_RPprev) and (D_RPprev < Dcrit_m)) else 0.0
                        RPmin_check_ok = 1.0 if (below_flag == 1.0 and np.isfinite(D_RPmin) and (D_RPmin >= Dcrit_m)) else 0.0

                        # FS3/FS4 diagnostics
                        if np.isfinite(N) and np.isfinite(A) and np.isfinite(B):
                            if np.isfinite(k_RPmin) and (A + k_RPmin * B) > 0:
                                FS3 = float(N / (A + k_RPmin * B))
                            if np.isfinite(k_RPprev) and (A + k_RPprev * B) > 0:
                                FS4 = float(N / (A + k_RPprev * B))

                else:
                    # default Grilli threshold mode
                    hit = np.where(k_series >= float(kcrit))[0]
                    if hit.size > 0:
                        jmin = int(hit[0])
                        RPmin = float(rp_ladder[jmin])
                        k_RPmin = float(k_series[jmin])
                        failed_within_cap = 1

                        RPmin_is_lb = 1 if abs(RPmin - float(rp_min)) <= 1e-12 else 0

                        if RPmin_is_lb:
                            RPmin_check_ok = np.nan
                            below_flag = np.nan
                            RPprev = float(rp_min)
                            k_RPprev = float(_haz_k_from_npz(npz, s_center, RPprev))
                        else:
                            RPprev = float(max(float(rp_min), RPmin - float(check_delta)))
                            k_RPprev = float(_haz_k_from_npz(npz, s_center, RPprev))
                            below_flag = 1.0 if (np.isfinite(k_RPprev) and (k_RPprev < float(kcrit))) else 0.0
                            RPmin_check_ok = 1.0 if (below_flag == 1.0 and np.isfinite(k_RPmin) and (k_RPmin >= float(kcrit))) else 0.0

                        # FS3/FS4 diagnostics
                        if np.isfinite(N) and np.isfinite(A) and np.isfinite(B):
                            if np.isfinite(k_RPmin) and (A + k_RPmin * B) > 0:
                                FS3 = float(N / (A + k_RPmin * B))
                            if np.isfinite(k_RPprev) and (A + k_RPprev * B) > 0:
                                FS4 = float(N / (A + k_RPprev * B))

        # depth for eta0: use REAL bathy at center
        zb_center_real = float(np.interp(s_center, s_seg, zb_seg_real))
        d_center = float(max(1.0, -zb_center_real))

        eta0 = eta0_rotational_eq19(
            beta_deg=beta_deg,
            T=T_guess,
            l=L_eff,
            d=d_center,
            r=r_circle,
            dphi=dphi,
            rho_s=rho_s,
            rng=rng
        )

        out_rows.append(dict(
            transect_id=int(tid),
            trial_id=int(len(out_rows)),
            s0_m=float(s0),
            s1_m=float(s1),
            L_eff_m=float(L_eff),
            d0_m=float(d0),
            beta_deg=float(beta_deg),
            rho_s=float(rho_s),
            su_ratio=float(su_ratio),
            Rur=float(Rur),
            rr_max=float(rr),
            R_gauss=float(Rgauss),
            T_guess_m=float(T_guess),
            sagitta_m=float(sagitta),
            T_max_m=float(T_max),
            r_m=float(r_circle),
            dphi_rad=float(dphi),
            d_center_m=float(d_center),
            FS0=float(FS0),
            kcrit=float(kcrit),

            RPmin_yr=float(RPmin) if np.isfinite(RPmin) else np.nan,
            k_RPmin=float(k_RPmin) if np.isfinite(k_RPmin) else np.nan,
            failed_within_cap=int(failed_within_cap),

            RPprev_yr=float(RPprev) if np.isfinite(RPprev) else np.nan,
            k_RPprev=float(k_RPprev) if np.isfinite(k_RPprev) else np.nan,

            RPmin_is_lower_bound=int(RPmin_is_lb),
            RPmin_check_ok=float(RPmin_check_ok) if np.isfinite(RPmin_check_ok) else np.nan,
            RPprev_is_below_rpmin=float(below_flag) if np.isfinite(below_flag) else np.nan,

            FS3=float(FS3) if np.isfinite(FS3) else np.nan,
            FS4=float(FS4) if np.isfinite(FS4) else np.nan,

            eta0_m=float(eta0) if np.isfinite(eta0) else np.nan,

            failure_mode=str(failure_mode),
            M_evt=float(M_evt) if np.isfinite(M_evt) else np.nan,
            Dcrit_m=float(Dcrit_m) if np.isfinite(Dcrit_m) else np.nan,
            D_RPmin_m=float(D_RPmin) if np.isfinite(D_RPmin) else np.nan,
            D_RPprev_m=float(D_RPprev) if np.isfinite(D_RPprev) else np.nan,
            newmark_eps_ln=float(newmark_eps_ln) if np.isfinite(newmark_eps_ln) else np.nan,

            failure_type="rot"
        ))

    out = pd.DataFrame(out_rows, columns=OUT_COLS)
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[OK] tr_{tid:04d}: {len(out)} accepted / {attempts} attempts -> {out_path}")
    return (tid, int(len(out)), int(attempts))


def main():
    ap = argparse.ArgumentParser(
        description="Rotational (slump) Monte Carlo per transect — Grilli-basics (plane slope in slide region).\n"
                    "Failure modes:\n"
                    "  - kcrit_threshold: fail when k(RP) >= kcrit (classic)\n"
                    "  - newmark_rs2009 : fail when D(RP) >= Dcrit, with D from Rathje & Saygili (2009) PGA+M\n"
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--transects-dir", default=None)
    ap.add_argument("--hazard-dir", default=None)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--out-fmt", default=None)

    ap.add_argument("--n-trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-fs-checks", action="store_true")

    ap.add_argument("--rp-min", type=float, default=100.0)
    ap.add_argument("--rp-max", type=float, default=750.0)
    ap.add_argument("--rp-step", type=float, default=10.0)
    ap.add_argument("--rp-cap", type=float, default=750.0)
    ap.add_argument("--check-delta", type=float, default=10.0)

    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--backend", type=str, default="loky", choices=["loky","multiprocessing","threading"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"]) if args.transects_dir is None else os.path.abspath(args.transects_dir)

    rot_param_path = abspath(cfg, cfg["paths"]["parameters"]["rot"]) if not args.params else os.path.abspath(args.params)
    user_params = _load_json(rot_param_path)
    params = _merge_defaults(user_params, DEFAULTS)

    hazard_dir = os.path.abspath(args.hazard_dir) if args.hazard_dir else os.path.join(transects_dir, "hazard_fits")
    hazard_map, idx_path = _load_hazard_index(hazard_dir)

    rp_min = float(args.rp_min)
    rp_max = float(args.rp_max)
    rp_step = float(args.rp_step)
    rp_cap = float(args.rp_cap)
    check_delta = float(args.check_delta)

    rp_ladder = [float(x) for x in np.arange(rp_min, rp_max + 0.5 * rp_step, rp_step) if x > 0]
    if np.isfinite(rp_cap) and rp_cap > 0:
        rp_ladder = [rp for rp in rp_ladder if rp <= rp_cap + 1e-12]

    pattern = args.pattern if args.pattern else os.path.join(transects_dir, "tr_*.csv")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if os.path.basename(f) != "index.csv" and "_mcs" not in os.path.basename(f) and "transects_over" not in os.path.basename(f)]
    if not files:
        raise SystemExit(f"No transect CSV files for pattern: {pattern}")

    out_fmt = args.out_fmt if args.out_fmt else os.path.join(transects_dir, "tr_{tid:04d}_grilli_rot_mcs.csv")

    fm = str((params.get("failure", {}) or {}).get("mode", "kcrit_threshold"))
    print(f"[INFO] rotational MC: {len(files)} transects | n_trials={args.n_trials} | n_jobs={args.n_jobs} | backend={args.backend}")
    print("[INFO] using parameters:", rot_param_path)
    print("[INFO] hazard_dir:", hazard_dir)
    if os.path.exists(idx_path):
        print("[INFO] hazard index:", idx_path)
    else:
        print("[WARN] hazard_fit_index.csv not found; will try canonical NPZ names tr_XXXXXX_pha_fit.npz")
    print(f"[INFO] RP ladder: min={rp_min} max={rp_max} step={rp_step} cap={rp_cap} check_delta={check_delta}yr")
    print("[INFO] keep_fs_checks:", bool(args.keep_fs_checks))
    print("[INFO] failure.mode:", fm)
    print("[INFO] outputs ->", out_fmt.format(tid=0).replace("0000", "????"))

    # Fallback: if hazard index doesn't list a file, try canonical name
    def _get_npz_path(tid_int: int) -> Optional[str]:
        p = hazard_map.get(tid_int, None)
        if p and os.path.exists(p):
            return p
        cand = os.path.join(hazard_dir, f"tr_{tid_int:06d}_pha_fit.npz")
        return cand if os.path.exists(cand) else None

    if args.n_jobs == 1:
        tot = 0
        for f in files:
            tid = int(parse_transect_id(f))
            npz_path = _get_npz_path(tid)
            _, nacc, _ = run_rotational_mc_for_transect(
                f, args.n_trials, args.seed, args.keep_fs_checks, out_fmt, params,
                npz_path, rp_ladder, rp_min, rp_cap, check_delta, verbose=True
            )
            tot += nacc
        print("[DONE] total accepted trials written:", tot)
        return

    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(run_rotational_mc_for_transect)(
            f, args.n_trials, args.seed, args.keep_fs_checks, out_fmt, params,
            _get_npz_path(int(parse_transect_id(f))),
            rp_ladder, rp_min, rp_cap, check_delta, args.verbose
        )
        for f in files
    )
    tot = int(sum(nacc for (_, nacc, _) in results))
    print("[DONE] total accepted trials written:", tot)


if __name__ == "__main__":
    main()