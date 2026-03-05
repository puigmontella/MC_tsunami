#!/usr/bin/env python3
"""
4_rotational_mc_areas.py

Rotational Monte Carlo per AREA tile, triggered by hazard curve fits (Step 3 areas).

Reads:
  - outputs/areas_dir/ar_XXXX.csv                (points, including x,y,depth,slope_deg)
  - outputs/areas_dir/hazard_fits/ar_XXXXXX_pga_fit.npz (or pha)
  - params: cfg["paths"]["parameters"]["rot"]

Writes:
  - outputs/areas_dir/ar_XXXX_rot_mcs.csv
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

from utilities.stats import trunc_normal, trunc_lognormal
from utilities.failure_tools import (
    sample_Rur,
    sample_Rur_direct,
    kcrit_rotational,
    eta0_rotational_predictive,
    newmark_disp_rathje_saygili_2009_m,
    make_circle_arc_through_endpoints,
    slip_on_transect,
    sample_bounded_lognormal,
    sample_thickness_T,
)

RHO_W = 1000.0

# placement defaults (same spirit as your translational areas script)
DEPTH_MATCH_TOL_M_DEFAULT = 200.0
MIN_CANDIDATES_DEFAULT = 20
FALLBACK_TOP_K_DEFAULT = 200

# local fit defaults
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
    "rho_s", "su_ratio",
    "Rur",
    "FS0", "kcrit",
    "RPmin_yr", "k_RPmin", "failed_within_cap",
    "T_m", "w_m", "d_center_m", "eta0_m", "V_m3", "d_misfit_m",
    "r_m", "dphi_rad",
]


# -----------------------------
# helpers
# -----------------------------
def parse_area_id(path):
    b = os.path.basename(path)
    m = re.search(r"ar_(\d+)", b)
    if not m:
        raise ValueError(f"Cannot parse area id from filename: {path}")
    return int(m.group(1))


def load_area_points(area_csv):
    """
    Expects: x,y,depth,slope_deg
    depth negative offshore.
    """
    df = pd.read_csv(area_csv)
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(df["depth"], errors="coerce").to_numpy(float)  # negative
    slope = pd.to_numeric(df["slope_deg"], errors="coerce").to_numpy(float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(slope)
    if np.count_nonzero(m) < 10:
        return None

    x = x[m]; y = y[m]; z = z[m]; slope = slope[m]
    depth_pos = -z
    return x, y, z, depth_pos, slope


def load_hazard_fit_area(hazard_dir, area_id):
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
    for k in ("ln_pga_coeffs", "lnpha_coeffs"):
        if k in z.files:
            coeff_key = k
            break
    if coeff_key is None:
        raise KeyError(f"Hazard NPZ missing coeffs. Keys={z.files}")

    coeffs = np.asarray(z[coeff_key], dtype=float)
    x = np.asarray(z["x"], dtype=float)
    y = np.asarray(z["y"], dtype=float)
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

    c = hz["coeffs"][:, j]            # lam^deg ... lam^0
    V = np.vander(lam, hz["deg"] + 1) # same ordering
    ln_k = V @ c
    k = np.exp(ln_k)
    k = np.maximum.accumulate(k)      # enforce monotonic in RP
    return k


def _plane_fit_beta_azimuth(xn, yn, zn, xc, yc, radius_m):
    """
    Weighted plane fit: z = a*x + b*y + c.
    beta from |grad z|; azimuth is downslope direction (toward decreasing z).
    Azimuth convention: degrees clockwise from North (0=N, 90=E).
    """
    X = np.c_[xn - xc, yn - yc, np.ones_like(xn)]
    z = zn

    r = np.sqrt((xn - xc)**2 + (yn - yc)**2)
    rw = max(1.0, 0.5 * radius_m)
    w = np.exp(-0.5 * (r / rw) ** 2)
    sw = np.sqrt(w)

    A = X * sw[:, None]
    bvec = z * sw

    try:
        p, _, rank, _ = np.linalg.lstsq(A, bvec, rcond=None)
    except Exception:
        return (np.nan, np.nan, False)

    if rank < 2 or not np.all(np.isfinite(p)):
        return (np.nan, np.nan, False)

    a, bcoef = float(p[0]), float(p[1])
    g = np.hypot(a, bcoef)
    if not np.isfinite(g) or g <= 1e-12:
        return (np.nan, np.nan, False)

    beta_deg = np.degrees(np.arctan(g))

    # downslope toward decreasing z => -grad
    vx, vy = (-a, -bcoef)  # East, North
    az = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0
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
            idx = np.where(dx*dx + dy*dy <= r*r)[0]

        if idx.size < FIT_MIN_PTS:
            # fallback to kNN
            if tree_pts is not None:
                k = min(FIT_KNN, xP.size)
                _, idx2 = tree_pts.query([xc, yc], k=k)
                idx = np.asarray(idx2, dtype=int).ravel()
            else:
                dx = xP - xc
                dy = yP - yc
                dd = dx*dx + dy*dy
                k = min(FIT_KNN, xP.size)
                idx = np.argpartition(dd, k-1)[:k]

        if idx.size >= FIT_MIN_PTS:
            beta_deg, az_deg, ok = _plane_fit_beta_azimuth(xP[idx], yP[idx], zP[idx], xc, yc, r)
            if ok and np.isfinite(beta_deg) and np.isfinite(az_deg):
                return float(beta_deg), float(az_deg)

    # fallback: direction to deepest neighbor
    if tree_pts is not None:
        k = min(80, xP.size)
        _, idx = tree_pts.query([xc, yc], k=k)
        idx = np.asarray(idx, dtype=int).ravel()
    else:
        dx = xP - xc
        dy = yP - yc
        dd = dx*dx + dy*dy
        k = min(80, xP.size)
        idx = np.argpartition(dd, k-1)[:k]

    idx = idx[idx != ic]
    if idx.size >= 1:
        j = idx[np.argmin(zP[idx])]  # most negative depth = deepest
        vx = float(xP[j] - xc)
        vy = float(yP[j] - yc)
        az = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0
        return np.nan, float(az)

    return np.nan, np.nan


# -----------------------------
# MC per area
# -----------------------------
def run_rot_mc_for_area(
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
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (aid, 0, 0)

    # params blocks (rotational.json)
    d0_cfg = params["d0"]
    L_cfg = params["L"]
    T_cfg = params["T"]
    rho_cfg = params["rho"]
    su_cfg = params["strength"]["su_ratio"]
    ru_cfg = params["ru"]
    geom_cfg = params.get("geometry", {}) or {}
    w_cfg = params.get("width", None)
    failure_cfg = params["failure"]

    # placement optional (defaults)
    pcfg = params.get("placement", {}) or {}
    depth_match_tol_m = float(pcfg.get("depth_match_tol_m", DEPTH_MATCH_TOL_M_DEFAULT))
    min_candidates = int(pcfg.get("min_candidates", MIN_CANDIDATES_DEFAULT))
    fallback_top_k = int(pcfg.get("fallback_top_k", FALLBACK_TOP_K_DEFAULT))

    # geometry controls
    arc_npts = int(geom_cfg.get("arc_npts", 200))
    sagitta_scale = float(geom_cfg.get("sagitta_scale", 1.0))
    max_attempts_factor = int(geom_cfg.get("max_attempts_factor", 30))

    # Ru controls
    eps_r = float(ru_cfg.get("eps_r", 0.35))
    r_gauss_std = float(ru_cfg.get("r_gauss_std", 0.25))
    clip_min = float(ru_cfg.get("clip_min", 0.0))
    clip_max = float(ru_cfg.get("clip_max", 0.95))

    # width optional
    if w_cfg is not None:
        w_min = float(w_cfg["w_min"])
        w_max = float(w_cfg["w_max"])
    else:
        w_min = w_max = np.nan

    failure_mode = str(failure_cfg["mode"]).lower().strip()
    if failure_mode not in ("kcrit_threshold", "newmark_rs2009"):
        raise RuntimeError(f"Unknown failure.mode='{failure_mode}'")

    sigma_ln_rs = float(failure_cfg.get("sigma_ln", 0.7))
    Dcrit_cfg = failure_cfg.get("Dcrit_m", {}) or {}

    rng = np.random.default_rng(int(seed + aid * 10007))

    out_rows = []
    attempts = 0
    max_attempts = int(n_trials) * int(max_attempts_factor)

    m_usable = np.isfinite(depth_pos_all) & np.isfinite(slope_deg_all) & (depth_pos_all > 0.0) & (slope_deg_all > 0.0)
    usable_idx = np.where(m_usable)[0]
    if usable_idx.size < 10:
        pd.DataFrame([], columns=OUT_COLS).to_csv(out_path, index=False)
        return (aid, 0, 0)

    rp_grid = np.asarray(rp_grid, dtype=float)
    rp_grid = np.unique(rp_grid)

    while (len(out_rows) < n_trials) and (attempts < max_attempts):
        attempts += 1

        # sample priors
        d0 = trunc_lognormal(rng, d0_cfg["mean"], d0_cfg["std"], d0_cfg["min"], d0_cfg["max"])
        L_eff = trunc_lognormal(rng, L_cfg["mean"], L_cfg["std"], L_cfg["min"], L_cfg["max"])
        rho_s = trunc_normal(rng, rho_cfg["mean"], rho_cfg["std"], rho_cfg["min"], rho_cfg["max"])
        su_ratio = trunc_normal(rng, su_cfg["mean"], su_cfg["std"], su_cfg["min"], su_cfg["max"])

        # placement: choose center point near d0 (same logic as trans areas)
        d_err = np.abs(depth_pos_all[usable_idx] - float(d0))
        tol = min(depth_match_tol_m, 0.35 * float(d0))
        cand_local = usable_idx[d_err <= tol]
        if cand_local.size >= min_candidates:
            ic = int(rng.choice(cand_local))
        else:
            k = min(fallback_top_k, usable_idx.size)
            top = usable_idx[np.argpartition(d_err, k - 1)[:k]]
            ic = int(rng.choice(top))

        x_center = float(xP[ic])
        y_center = float(yP[ic])
        z_center = float(zP[ic])           # negative
        d_center = float(depth_pos_all[ic])
        beta_pt_deg = float(slope_deg_all[ic])

        # local mean slope + azimuth (downslope)
        beta_mean_deg, azimuth_deg = fit_local_beta_mean_and_azimuth(
            xP, yP, zP, ic, L_eff, tree_pts=tree_pts
        )

        beta_use = beta_mean_deg if np.isfinite(beta_mean_deg) else beta_pt_deg
        if not np.isfinite(beta_use) or beta_use <= 0.0:
            continue

        if not np.isfinite(azimuth_deg):
            azimuth_deg = float(rng.uniform(0.0, 360.0))

        # Ru sampling
        ru_model = str(ru_cfg["model"]).lower().strip()
        if ru_model == "grilli":
            Rur, _, _ = sample_Rur(beta_use, rng, eps_r=eps_r, r_gauss_std=r_gauss_std)
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

        # thickness
        T = sample_thickness_T(rng, T_cfg, L_eff)
        sagitta = float(max(1e-6, sagitta_scale * T))

        # build a synthetic 1D seabed profile along downslope with constant beta_use:
        g = float(np.tan(np.radians(beta_use)))  # |dz/ds|
        n_nodes = max(60, int(L_eff / 25.0))     # ~25 m resolution, clamp min
        s_seg = np.linspace(0.0, float(L_eff), int(n_nodes)).astype(float)
        zb_seg = (z_center - g * (s_seg - 0.5 * float(L_eff))).astype(float)  # negative depths

        # endpoints
        s0 = float(s_seg[0]);  z0 = float(zb_seg[0])
        s1 = float(s_seg[-1]); z1 = float(zb_seg[-1])

        arc = make_circle_arc_through_endpoints(s0, z0, s1, z1, sagitta=sagitta, npts=arc_npts)
        if arc is None:
            continue

        s_arc, z_arc, r_circle, dphi, _ = arc
        if (not np.isfinite(r_circle)) or (r_circle <= 0.0):
            continue

        dphi_use = float(dphi)
        if np.isfinite(dphi_use) and (dphi_use > 2.0 * np.pi + 1e-6):
            dphi_use = float(np.deg2rad(dphi_use))
        if (not np.isfinite(dphi_use)) or (dphi_use <= 0.0):
            continue

        zs_seg = slip_on_transect(s_seg, s_arc, z_arc)
        th = zb_seg - zs_seg
        if np.any(~np.isfinite(th)) or np.any(th <= 0.0):
            continue

        # stability
        FS0, kcrit = kcrit_rotational(s_seg, zb_seg, zs_seg, rho_s, su_ratio, Rur, r_circle)
        if not np.isfinite(FS0) or not np.isfinite(kcrit):
            continue
        if keep_fs_checks and (FS0 <= 1.0):
            continue

        # optional width
        if np.isfinite(w_min) and np.isfinite(w_max):
            w = float(rng.uniform(w_min, w_max)) * float(L_eff)
        else:
            w = np.nan

        # eta0 proxy
        eta0 = eta0_rotational_predictive(
            beta_deg=beta_use,
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

        # hazard at center
        k_series = eval_k_at_xy(hz, x_center, y_center, rp_grid)

        RPmin = np.nan
        k_RPmin = np.nan
        failed_within_cap = 0

        if failure_mode == "newmark_rs2009":
            if magValues is None or len(magValues) == 0:
                raise RuntimeError("failure.mode=newmark_rs2009 requires magnitude.csv")
            M_evt = float(rng.choice(magValues))

            Dcrit_m = sample_bounded_lognormal(
                rng,
                Dcrit_cfg["median"], Dcrit_cfg["sigma_ln"],
                Dcrit_cfg["min"], Dcrit_cfg["max"]
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

            "rho_s": round(float(rho_s), 1),
            "su_ratio": round(float(su_ratio), 4),

            "Rur": round(float(Rur), 3),

            "FS0": round(float(FS0), 2),
            "kcrit": round(float(kcrit), 6),

            "RPmin_yr": int(RPmin) if np.isfinite(RPmin) else np.nan,
            "k_RPmin": round(float(k_RPmin), 6) if np.isfinite(k_RPmin) else np.nan,
            "failed_within_cap": int(failed_within_cap),

            "T_m": round(float(T), 1),
            "w_m": round(float(w), 1) if np.isfinite(w) else np.nan,
            "d_center_m": round(float(d_center), 1),
            "eta0_m": round(float(eta0), 3),

            "V_m3": int(float(L_eff) * float(w) * float(T)) if np.isfinite(w) else np.nan,
            "d_misfit_m": int(abs(float(d_center) - float(d0))),

            "r_m": round(float(r_circle), 1),
            "dphi_rad": round(float(dphi_use), 6),
        })

    out = pd.DataFrame(out_rows, columns=OUT_COLS)
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[OK] ar_{aid:04d}: {len(out_rows)} accepted / {attempts} attempts -> {out_path}")

    return (aid, int(len(out_rows)), int(attempts))


def main():
    ap = argparse.ArgumentParser(description="Rotational Monte Carlo per AREA (hazard-triggered).")
    ap.add_argument("--config", default="config.json")
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
    if "paths" not in cfg or "parameters" not in cfg["paths"] or "rot" not in cfg["paths"]["parameters"]:
        raise SystemExit("[FAIL] config missing: paths.parameters.rot")

    areas_dir = cfg["outputs"]["areas_dir"]
    hazard_dir = os.path.join(areas_dir, "hazard_fits")
    if not os.path.isdir(hazard_dir):
        raise SystemExit(f"[FAIL] Missing hazard_dir: {hazard_dir}")

    params_path = cfg["paths"]["parameters"]["rot"]
    with open(params_path, "r") as f:
        params = json.load(f)

    # failure mode + magnitude list (optional)
    failure_mode = str(params["failure"]["mode"])
    if failure_mode == "newmark_rs2009":
        mag_path = cfg["paths"]["parameters"]["magnitude_csv"]
        if not os.path.exists(mag_path):
            raise SystemExit(f"[FAIL] Missing magnitude CSV: {mag_path}")
        df_M = pd.read_csv(mag_path)
        magValues = pd.to_numeric(df_M["M_evt"], errors="coerce").to_numpy(float)
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

    out_fmt = os.path.join(areas_dir, "ar_{aid:04d}_rot_mcs.csv")

    print(f"[INFO] rotational MC (AREAS): {len(files)} areas")
    print(f"[INFO] n_trials={args.n_trials}  n_jobs={args.n_jobs}  backend={args.backend}")
    print("[INFO] params:", params_path)
    print("[INFO] hazard_dir:", hazard_dir)
    print(f"[INFO] RP ramp: min={rp_min} max={rp_max} step={rp_step}")
    print("[INFO] keep_fs_checks:", bool(args.keep_fs_checks))
    print("[INFO] failure.mode:", failure_mode)

    if args.n_jobs == 1:
        tot = 0
        for fcsv in files:
            _, nacc, _ = run_rot_mc_for_area(
                fcsv, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
                params, hazard_dir, rp_grid, magValues, verbose=True
            )
            tot += nacc
        print("[DONE] total accepted trials written:", tot)
        return

    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(run_rot_mc_for_area)(
            fcsv, args.n_trials, args.seed, args.keep_fs_checks, out_fmt,
            params, hazard_dir, rp_grid, magValues, args.verbose
        )
        for fcsv in files
    )
    tot = int(sum(nacc for (_, nacc, _) in results))
    print("[DONE] total accepted trials written:", tot)


if __name__ == "__main__":
    main()