#!/usr/bin/env python3
"""
3_loading_PGA.py

Sample PGA (or PHA) rasters along each transect and build a simple per-point hazard curve fit.

What this step does
-------------------
For each transect (made in Step 2):
  1) read the transect points from outputs/transects_dir/tr_XXXX.csv
  2) sample each PGA raster (one per return period) at those (x,y)
  3) fit: ln(PGA_g) = polynomial(lambda), where lambda = 1 / RP_years (annual exceedance rate)
  4) save coefficients + sampled values to a .npz file

Run
-------------------------
  python ../../Python/3_loading_PGA.py 

  to run it in parallel (with 30 cores):
  python ../../Python/3_loading_PGA.py --n-jobs 30 

Outputs
-------
By default writes to:
  outputs/hazard_fits/
    tr_000000_pga_fit.npz
    tr_000001_pga_fit.npz
    ...
    hazard_fit_index.csv

Help
----
python ../../Python/3_loading_PGA.py -h

Notes
-----
- With only 4 return periods, avoid high-degree polynomials. Default here is degree=2.
"""


import os
import json
import argparse
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed


# -----------------------------
# functions
# -----------------------------

def sample_raster_points(src, xs, ys):
    """
    Sample raster values at (xs, ys) using rasterio
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    pts = list(zip(xs, ys))
    vals = np.array([v[0] for v in src.sample(pts)], dtype=float)
    return vals


def fit_ln_pga_poly(lambda_exceed, ln_pga, deg):
    """
    Fit ln(PGA) = poly(lambda) point-by-point.
    """
    lam = np.asarray(lambda_exceed, dtype=float)
    Y = np.asarray(ln_pga, dtype=float)

    n_rp, n_pts = Y.shape
    coeffs = np.full((deg + 1, n_pts), np.nan, dtype=float)

    V = np.vander(lam, deg + 1)

    for j in range(n_pts):
        y = Y[:, j]
        m = np.isfinite(lam) & np.isfinite(y)
        if np.sum(m) < (deg + 1):
            continue
        c, *_ = np.linalg.lstsq(V[m, :], y[m], rcond=None)
        coeffs[:, j] = c

    return coeffs


def process_one_transect(transect_id, transects_dir, rp_list, pga_tifs, deg, out_dir):
    """
    Read one transect CSV, sample PGA rasters, fit ln(PGA) curve, save as NPZ.
    """
    tr_csv = transects_dir + f"/tr_{int(transect_id):04d}.csv"
    if not os.path.exists(tr_csv):
        return dict(transect_id=int(transect_id), n_pts=0, out_npz="", status="missing_transect_csv")

    df = pd.read_csv(tr_csv)

    s = df["s_m"].to_numpy(float)
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    n_pts = len(s)
    n_rp = len(rp_list)

    rp_years = np.array([float(rp) for rp in rp_list], dtype=float)
    lambda_exceed = 1.0 / rp_years

    pga = np.full((n_rp, n_pts), np.nan, dtype=float)

    # sample each RP raster (open -> sample -> close)
    for i, rp in enumerate(rp_list):
        tif = pga_tifs[str(rp)]
        with rasterio.open(tif) as src:
            vals = sample_raster_points(src, x, y)
        pga[i, :] = vals

    ln_pga = np.log(np.clip(pga, 1e-12, np.inf))
    coeffs = fit_ln_pga_poly(lambda_exceed, ln_pga, deg=deg)

    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"tr_{int(transect_id):06d}_pga_fit.npz")
    s_arr = np.asarray(s)
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    np.savez_compressed(
        out_npz,
        transect_id=int(transect_id),
        s_m=np.round(s_arr, 1).astype(np.float32),   # keep 0.1 m resolution
        x=np.rint(x_arr).astype(np.int32),
        y=np.rint(y_arr).astype(np.int32),
        rp_years=np.asarray(rp_years, dtype=np.float32),
        lambda_exceed=np.asarray(lambda_exceed, dtype=np.float32),
        deg=int(deg),
        ln_pga_coeffs=np.asarray(coeffs, dtype=np.float64),
        pga_samples=np.asarray(pga, dtype=np.float32),
    )

    return dict(transect_id=int(transect_id), n_pts=int(n_pts), out_npz=out_npz, status="ok")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 3: sample PGA rasters along transects and fit hazard curves.")
    ap.add_argument("--config", default="config.json", help="Path to case config.json (default: config.json)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel cores (-1 = all cores)")
    ap.add_argument("--backend", type=str, default="threading", choices=["threading", "loky"])
    ap.add_argument("--deg", type=int, default=2, help="Polynomial degree for ln(PGA) fit (default: 2).")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    transects_dir = cfg["outputs"]["transects_dir"]
    index_csv = os.path.join(transects_dir, "index.csv")
    if not os.path.exists(index_csv):
        raise SystemExit(f"[FAIL] Missing index.csv: {index_csv}")

    idx = pd.read_csv(index_csv)
    if "id" not in idx.columns:
        raise SystemExit(f"[FAIL] index.csv missing 'id'. Columns: {list(idx.columns)}")
    transect_ids = idx["id"].astype(int).tolist()

    pga_dict = cfg["paths"]["pga_rasters"]  # {"475": "...tif", ...}
    rp_list = sorted(pga_dict.keys(), key=float)

    if len(rp_list) < 2:
        raise SystemExit("[FAIL] Need at least 2 return periods to build a curve.")

    # clamp degree so it always makes sense
    deg = max(1, min(int(args.deg), len(rp_list) - 1))


    pga_tifs = {}
    for rp in rp_list:
        tif = pga_dict[rp]
        if not os.path.exists(tif):
            raise SystemExit(f"[FAIL] Missing PGA raster for RP={rp}: {tif}")
        pga_tifs[str(rp)] = tif

    out_dir = transects_dir + "/hazard_fits"
    os.makedirs(out_dir, exist_ok=True)

    prefer = "threads" if args.backend == "threading" else "processes"
    rows = Parallel(n_jobs=args.n_jobs, backend=args.backend, prefer=prefer, verbose=10)(
        delayed(process_one_transect)(
            transect_id=tid,
            transects_dir=transects_dir,
            rp_list=rp_list,
            pga_tifs=pga_tifs,
            deg=deg,
            out_dir=out_dir,
        )
        for tid in transect_ids
    )

    # out_index = out_dir + "/hazard_fit_index.csv"
    # pd.DataFrame(rows).to_csv(out_index, index=False)
    # print(f"[OK] wrote {out_index}")
    print(f"[OK] degree used: {deg} (n_rp={len(rp_list)})")


if __name__ == "__main__":
    main()