#!/usr/bin/env python3
"""
4_translational_mc_areas.py

Translational Monte Carlo per AREA tile, triggered by a hazard curve fit.

Fixes / robustness:
- Hazard fit loader is flexible on coeff key (ln_pga_coeffs / lnpha_coeffs / ln_pha_coeffs).
- If hazard fit missing, writes empty output instead of crashing (pipeline-friendly).
- failure.mode and ru.model treated case-insensitively.
- Optional params["beta_mode"] to control beta used in calculations:
    "auto" (default): beta_mean if finite else beta_pt
    "pt"            : always beta_pt_deg
    "mean"          : beta_mean if finite else beta_pt_deg
    "max" / "min"   : combine mean and pt (debug)
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utilities.stats import trunc_normal, trunc_lognormal
from utilities.failure_tools import (
    sample_Rut,
    sample_Rut_direct,
    kcrit_translational,
    eta0_translational_predictive,
    newmark_disp_rathje_saygili_2009_m,
    sample_bounded_lognormal,
    sample_thickness_T,
)

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

RHO_W = 1000.0  # kg/m^3

# Defaults (so you don't need many JSON knobs)
DEPTH_MATCH_TOL_M_DEFAULT = 200.0
MIN_CANDIDATES_DEFAULT = 20
FALLBACK_TOP_K_DEFAULT = 200

# Local fit defaults
FIT_MIN_PTS = 30
FIT_KNN = 120
FIT_R0_MIN = 500.0
FIT_R_MAX = 6000.0
FIT_R_MULTS = (1.0, 1.6, 2.5, 4.0)

OUT_COLS = [
    "area_id", "trial_id",
    "x_center_m", "y_center_m",
    "d0_m", "L_eff_m",
    "beta_pt_deg", "beta_mean_deg", "azimuth_deg",
    "phi_deg", "rho_s",
    "Rut", "FS2", "kcrit",
    "RPmin_yr", "k_RPmin", "failed_within_cap",
    "T_m", "w_m", "d_center_m", "eta0_m", "V_m3", "d_misfit_m",
]


# -----------------------------
# Helpers
# -----------------------------
def parse_area_id(path):
    b = os.path.basename(path)
    m = re.search(r"ar_(\d+)", b)
    if not m:
        raise ValueError(f"Cannot parse area id from filename: {path}")
    return int(m.group(1))


def load_area_points(area_csv):
    """
    Read one area CSV. Expects columns: x,y,depth,slope_deg
    depth is negative offshore.
    """
    df = pd.read_csv(area_csv)

    x = pd.to_numeric(df.get("x"), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.get("y"), errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df.get("depth"), errors="coerce").to_numpy(dtype=float)  # negative offshore
    slope = pd.to_numeric(df.get("slope_deg"), errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(slope)
    if np.count_nonzero(m) < 10:
        return None

    x = x[m]
    y = y[m]
    z = z[m]
    slope = slope[m]

    depth_pos = -z  # positive
    return x, y, z, depth_pos, slope


def load_hazard_fit_area(hazard_dir, area_id):
    """
    Accepts either ar_XXXXXX_pga_fit.npz or ar_XXXXXX_pha_fit.npz.
    Accept coeff keys: ln_pga_coeffs, lnpha_coeffs, ln_pha_coeffs
    Returns None if missing.
    """
    cand = [
        os.path.join(hazard_dir, f"ar_{area_id:06d}_pga_fit.npz"),
        os.path.join(hazard_dir, f"ar_{area_id:06d}_pha_fit.npz"),
    ]
    f = None
    for c in cand:
        if os.path.exists(c):
            f = c
            break
    if f is None:
        return None

    z = np.load(f, allow_pickle=False)

    coeff_key = None
    for k in ("ln_pga_coeffs", "lnpha_coeffs", "ln_pha_coeffs"):
        if k in z.files:
            coeff_key = k
            break
    if coeff_key is None:
        raise KeyError(f"[FAIL] Hazard NPZ missing coeffs. path={f} keys={z.files}")

    coeffs = np.asarray(z[coeff_key], dtype=float)
    x = np.asarray(z["x"], dtype=float)
    y = np.asarray(z["y"], dtype=float)

    if coeffs.ndim != 2 or x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"[FAIL] Bad hazard fit shapes in {f}")
    if coeffs.shape[1] != x.size or x.size != y.size:
        raise ValueError(f"[FAIL] Hazard fit arrays inconsistent in {f}")

    deg = int(z["deg"]) if "deg" in z.files else (coeffs.shape[0] - 1)

    hz = dict(npz_path=f, deg=deg, coeffs=coeffs, x=x, y=y, tree=None)
    if cKDTree is not None:
        hz["tree"] = cKDTree(np.c_[x, y])
    return hz


def eval_k_at_xy(hz, x0, y0, rp_grid):
    rp = np.asarray(rp_grid, dtype=float)
    lam = 1.0 / rp

    if hz["tree"] is not None:
        _, j = hz["tree"].query([float(x0), float(y0)], k=1)
        j = int(j)
    else:
        dx = hz["x"] - float(x0)
        dy = hz["y"] - float(y0)
        j = int(np.argmin(dx * dx + dy * dy))

    c = hz["coeffs"][:, j]
    V = np.vander(lam, hz["deg"] + 1)  # lam^deg ... lam^0
    ln_k = V @ c
    k = np.exp(ln_k)
    k = np.maximum.accumulate(k)  # enforce monotonic
    return k


def _plane_fit_beta_azimuth(xn, yn, zn, xc, yc, radius_m):
    """
    Weighted plane fit: z = a*(x-xc) + b*(y-yc) + c
    Returns (beta_deg, azimuth_deg, ok).
    Azimuth convention: degrees clockwise from North (0=N, 90=E).
    """
    X = np.c_[xn - xc, yn - yc, np.ones_like(xn)]
    z = zn

    r = np.sqrt((xn - xc) ** 2 + (yn - yc) ** 2)
    rw = max(1.0, 0.5 * radius_m)
    w = np.exp(-0.5 * (r / rw) ** 2)
    sw = np.sqrt(w)

    A = X * sw[:, None]
    b = z * sw

    try:
        p, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return (np.nan, np.nan, False)

    if rank < 2 or not np.all(np.isfinite(p)):
        return (np.nan, np.nan, False)

    a, bcoef = float(p[0]), float(p[1])
    g = np.hypot(a, bcoef)
    if not np.isfinite(g) or g <= 1e-12:
        return (np.nan, np.nan, False)

    beta_deg = np.degrees(np.arctan(g))

    # downslope points toward decreasing z => -grad
    vx, vy = (-a, -bcoef)  # East, North
    az = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0  # atan2(E,N)
    return (beta_deg, az, True)


def fit_local_beta_mean_and_azimuth(xP, yP, zP, ic, L_eff, tree_pts=None):
    xc = float(xP[ic]); yc = float(yP[ic])

    r0 = max(FIT_R0_MIN, 0.35 * float(L_eff))
    radii = [min(FIT_R_MAX, r0 * m) for m in FIT_R_MULTS]

    for r in radii:
        if tree_pts is not None:
            idx = np.asarray(tree_pts.query_ball_point([xc, yc], r=r), dtype=int)
        else:
            dx = xP - xc
            dy = yP - yc
            idx = np.where(dx * dx + dy * dy <= r * r)[0]

        if idx.size < FIT_MIN_PTS:
            if tree_pts is not None:
                k = min(FIT_KNN, xP.size)
                _, idx2 = tree_pts.query([xc, yc], k=k)
                idx = np.asarray(idx2, dtype=int).ravel()
            else:
                dx = xP - xc
                dy = yP - yc
                dd = dx * dx + dy * dy
                k = min(FIT_KNN, xP.size)
                idx = np.argpartition(dd, k - 1)[:k]

        if idx.size >= FIT_MIN_PTS:
            beta_deg, az_deg, ok = _plane_fit_beta_azimuth(xP[idx], yP[idx], zP[idx], xc, yc, r)
            if ok and np.isfinite(beta_deg) and np.isfinite(az_deg):
                return float(beta_deg), float(az_deg)

    # Fallback: direction to deepest neighbor among kNN
    if tree_pts is not None:
        k = min(80, xP.size)
        _, idx = tree_pts.query([xc, yc], k=k)
        idx = np.asarray(idx, dtype=int).ravel()
    else:
        dx = xP - xc
        dy = yP - yc
        dd = dx * dx + dy * dy
        k = min(80, xP.size)
        idx = np.argpartition(dd, k - 1)[:k]

    idx = idx[idx != ic]
    if idx.size >= 1:
        j = idx[np.argmin(zP[idx])]  # most negative => deepest
        vx = float(xP[j] - xc)
        vy = float(yP[j] - yc)
        az = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0
        return np.nan, float(az)

    return np.nan, np.nan


def choose_beta(beta_pt_deg, beta_mean_deg, beta_mode):
    """
    beta_mode: auto|pt|mean|max|min
    """
    bp = float(beta_pt_deg) if np.isfinite(beta_pt_deg) else np.nan
    bm = float(beta_mean_deg) if np.isfinite(beta_mean_deg) else np.nan
    m = str(beta_mode).strip().lower()

    if m == "pt":
        return bp
    if m == "mean":
        return bm if np.isfinite(bm) else bp
    if m == "max":
        vals = [v for v in (bp, bm) if np.isfinite(v)]
        return max(vals) if vals else np.nan
    if m == "min":
        vals = [v for v in (bp, bm) if np.isfinite(v)]
        return min(vals) if vals else np.nan

    # default auto
    return bm if np.isfinite(bm) else bp


# -----------------------------
# Monte Carlo per area
# -----------------------------
def run_trans_mc_for_area(
    area_csv,
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
    aid = int(parse_area_id(area_csv))
    out_path = out_fmt.format(aid=aid)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pts = load_area_points(area_csv)
    if pts is None:
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (aid, 0, 0)

    xP, yP, zP, depth_pos_all, slope_deg_all = pts
    tree_pts = cKDTree(np.c_[xP, yP]) if (cKDTree is not None) else None

    hz = load_hazard_fit_area(hazard_dir, aid)
    if hz is None:
        # pipeline-friendly: write empty output instead of crashing
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (aid, 0, 0)

    d0_cfg = params["d0"]
    L_cfg = params["L"]
    T_cfg = params["T"]
    phi_cfg = params["phi"]
    rho_cfg = params["rho"]
    ru_cfg = params["ru"]
    w_cfg = params["width"]
    failure_cfg = params["failure"]

    # Optional placement block
    pcfg = params.get("placement", {}) or {}
    depth_match_tol_m = float(pcfg.get("depth_match_tol_m", DEPTH_MATCH_TOL_M_DEFAULT))
    min_candidates = int(pcfg.get("min_candidates", MIN_CANDIDATES_DEFAULT))
    fallback_top_k = int(pcfg.get("fallback_top_k", FALLBACK_TOP_K_DEFAULT))

    beta_mode = str(params.get("beta_mode", "auto"))

    failure_mode = str(failure_cfg["mode"]).strip().lower()
    if failure_mode not in ("kcrit_threshold", "newmark_rs2009"):
        raise RuntimeError(f"Unknown failure.mode='{failure_cfg['mode']}'")

    sigma_ln_rs = float(failure_cfg.get("sigma_ln", 0.7))
    Dcrit_cfg = failure_cfg.get("Dcrit_m", {}) or {}

    w_min = float(w_cfg["w_min"])
    w_max = float(w_cfg["w_max"])

    ru_model = str(ru_cfg["model"]).strip().lower()

    rng = np.random.default_rng(int(seed + aid * 10007))

    out_rows = []
    attempts = 0
    max_attempts = int(max(10 * n_trials, n_trials))

    m_usable = (
        np.isfinite(depth_pos_all) & np.isfinite(slope_deg_all) &
        (depth_pos_all > 0.0) & (slope_deg_all > 0.0)
    )
    usable_idx = np.where(m_usable)[0]
    if usable_idx.size < 10:
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (aid, 0, 0)

    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        d0 = trunc_lognormal(rng, d0_cfg["mean"], d0_cfg["std"], d0_cfg["min"], d0_cfg["max"])
        L = trunc_lognormal(rng, L_cfg["mean"], L_cfg["std"], L_cfg["min"], L_cfg["max"])
        phi_deg = trunc_normal(rng, phi_cfg["mean"], phi_cfg["std"], phi_cfg["min"], phi_cfg["max"])
        rho_s = trunc_normal(rng, rho_cfg["mean"], rho_cfg["std"], rho_cfg["min"], rho_cfg["max"])

        # Placement: choose center point where d_center ~ d0
        d_err = np.abs(depth_pos_all[usable_idx] - float(d0))
        tol = min(depth_match_tol_m, 0.35 * float(d0))
        cand_local = usable_idx[d_err <= tol]

        if cand_local.size >= min_candidates:
            ic = int(rng.choice(cand_local))
        else:
            k = min(fallback_top_k, usable_idx.size)
            if k < 1:
                continue
            top = usable_idx[np.argpartition(d_err, k - 1)[:k]]
            ic = int(rng.choice(top))

        x_center = float(xP[ic])
        y_center = float(yP[ic])
        d_center = float(depth_pos_all[ic])
        beta_pt_deg = float(slope_deg_all[ic])

        L_eff = float(L)
        if not np.isfinite(L_eff) or L_eff <= 0.0:
            continue

        T = sample_thickness_T(rng, T_cfg, L_eff)

        beta_mean_deg, azimuth_deg = fit_local_beta_mean_and_azimuth(
            xP, yP, zP, ic, L_eff, tree_pts=tree_pts
        )

        beta_use = choose_beta(beta_pt_deg, beta_mean_deg, beta_mode)
        if not np.isfinite(beta_use) or beta_use <= 0.0:
            continue

        if not np.isfinite(azimuth_deg):
            azimuth_deg = float(rng.uniform(0.0, 360.0))

        # Ru
        if ru_model == "grilli":
            Rut, *_ = sample_Rut(
                beta_deg=beta_use, phi_deg=phi_deg, rng=rng,
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
            raise ValueError(f"Unknown ru.model='{ru_cfg['model']}'")

        # FS checks
        tb = np.tan(np.radians(beta_use))
        tp = np.tan(np.radians(phi_deg))
        if tb <= 0.0 or (not np.isfinite(tb)) or (not np.isfinite(tp)):
            continue

        FS1 = tp / tb
        FS2 = (1.0 - float(Rut)) * FS1
        if keep_fs_checks and ((FS1 <= 1.0) or (FS2 <= 1.0)):
            continue

        kcrit = kcrit_translational(beta_use, phi_deg, rho_s, Rut)
        if not np.isfinite(kcrit):
            continue

        # width
        w = float(rng.uniform(w_min, w_max)) * L_eff
        if not np.isfinite(w) or w <= 0.0:
            continue

        # eta0
        eta0 = eta0_translational_predictive(beta_use, T, L_eff, d_center, rho_s, w, RHO_W)
        if not np.isfinite(eta0):
            continue

        # hazard at (x,y)
        k_series = eval_k_at_xy(hz, x_center, y_center, rp_grid)

        RPmin = np.nan
        k_RPmin = np.nan
        failed_within_cap = 0

        if failure_mode == "newmark_rs2009":
            if magValues is None or len(magValues) == 0:
                raise RuntimeError("failure.mode=newmark_rs2009 requires magnitude CSV.")

            M_evt = float(rng.choice(magValues))
            Dcrit_m = sample_bounded_lognormal(
                rng,
                float(Dcrit_cfg["median"]), float(Dcrit_cfg["sigma_ln"]),
                float(Dcrit_cfg["min"]), float(Dcrit_cfg["max"]),
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
            "area_id": int(aid),
            "trial_id": int(len(out_rows)),

            "x_center_m": int(round(x_center, 0)),
            "y_center_m": int(round(y_center, 0)),

            "d0_m": round(float(d0), 1),
            "L_eff_m": round(float(L_eff), 1),

            "beta_pt_deg": round(float(beta_pt_deg), 2),
            "beta_mean_deg": round(float(beta_mean_deg), 2) if np.isfinite(beta_mean_deg) else np.nan,
            "azimuth_deg": round(float(azimuth_deg), 1),

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
            "V_m3": int(L_eff * w * T),
            "d_misfit_m": int(abs(d_center - d0)),
        })

    out = pd.DataFrame(out_rows, columns=OUT_COLS)
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[OK] ar_{aid:04d}: {len(out_rows)} accepted / {attempts} attempts -> {out_path}")

    return (aid, int(len(out_rows)), int(attempts))


def main():
    ap = argparse.ArgumentParser(description="Translational Monte Carlo per AREA (hazard-triggered).")
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

    if "outputs" not in cfg or "areas_dir" not in cfg["outputs"]:
        raise SystemExit("[FAIL] config missing: outputs.areas_dir")
    if "paths" not in cfg or "parameters" not in cfg["paths"] or "trans" not in cfg["paths"]["parameters"]:
        raise SystemExit("[FAIL] config missing: paths.parameters.trans")

    areas_dir = cfg["outputs"]["areas_dir"]
    hazard_dir = os.path.join(areas_dir, "hazard_fits")
    if not os.path.isdir(hazard_dir):
        raise SystemExit(f"[FAIL] Missing hazard_dir: {hazard_dir}")

    params_path = cfg["paths"]["parameters"]["trans"]
    with open(params_path, "r") as f:
        params = json.load(f)

    failure_mode = str(params["failure"]["mode"]).strip().lower()
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

    files = sorted(glob.glob(os.path.join(areas_dir, "ar_*.csv")))
    files = [f for f in files if os.path.basename(f) != "index.csv" and "_mcs" not in os.path.basename(f)]
    if not files:
        raise SystemExit(f"[FAIL] No area CSV files found in: {areas_dir}")

    out_fmt = os.path.join(areas_dir, "ar_{aid:04d}_trans_mcs.csv")

    print(f"[INFO] translational MC (AREAS): {len(files)} areas")
    print(f"[INFO] n_trials={args.n_trials}  n_jobs={args.n_jobs}  backend={args.backend}")
    print("[INFO] params:", params_path)
    print("[INFO] hazard_dir:", hazard_dir)
    print(f"[INFO] RP ramp: min={rp_min} max={rp_max} step={rp_step}")
    print("[INFO] keep_fs_checks:", bool(args.keep_fs_checks))
    print("[INFO] failure.mode:", failure_mode)

    if args.n_jobs == 1:
        tot = 0
        for fcsv in files:
            _, nacc, _ = run_trans_mc_for_area(
                fcsv, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
                params, hazard_dir, rp_grid, magValues, verbose=True
            )
            tot += nacc
        print("[DONE] total accepted trials written:", tot)
        return

    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(run_trans_mc_for_area)(
            fcsv, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
            params, hazard_dir, rp_grid, magValues, args.verbose
        )
        for fcsv in files
    )
    tot = int(sum(nacc for (_, nacc, _) in results))
    print("[DONE] total accepted trials written:", tot)


if __name__ == "__main__":
    main()