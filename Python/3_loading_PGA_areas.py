#!/usr/bin/env python3
"""
3_loading_PGA_areas.py

Sample PGA (or PHA) rasters at points inside each AREA tile (Step 2) and fit a simple
per-point hazard curve:

  ln(PGA_g) = polynomial(lambda), where lambda = 1 / RP_years (annual exceedance rate)

What this step does
-------------------
For each area (made in Step 2):
  1) read the sampled points from outputs/areas_dir/ar_XXXX.csv
  2) sample each PGA raster (one per return period) at those (x,y)
  3) fit ln(PGA) vs lambda with a low-degree polynomial (default deg=2, clamped)
  4) save coefficients + sampled values to an .npz file

Run
---
  python ../../Python/3_loading_PGA_areas.py
  python ../../Python/3_loading_PGA_areas.py --n-jobs 30
  python ../../Python/3_loading_PGA_areas.py --plot-one --area-id 12   (optional quick check)

Outputs
-------
Writes to:
  outputs/areas_dir/hazard_fits/
    ar_000000_pga_fit.npz
    ar_000001_pga_fit.npz
    ...
    hazard_fit_index.csv

Notes
-----
- With only 4 return periods, avoid high-degree polynomials. Default is deg=2, clamped to <= n_rp-1.
- Default backend is threading (recommended here). If you switch to loky (processes),
  the script will avoid pre-loading rasters to prevent huge inter-process copies.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed

from utilities.geoio import (
    load_raster_band,
    sample_loaded_raster,
)

# -----------------------------
# helpers
# -----------------------------
def _nanify_nodata(vals, nodata):
    if nodata is None:
        return vals
    if isinstance(nodata, float) and np.isnan(nodata):
        return vals
    return np.where(vals == nodata, np.nan, vals)


def sample_raster_points_open(src, xs, ys):
    """
    Sample raster values at (xs, ys) using rasterio.open handle (no pre-load).
    Returns float64 with nodata -> NaN.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    pts = list(zip(xs, ys))
    vals = np.array([v[0] for v in src.sample(pts)], dtype="float64")
    return _nanify_nodata(vals, src.nodata)


def fit_ln_pga_poly_vectorized(lambda_exceed, ln_pga, deg):
    """
    Vectorized fit for columns with complete data:
      ln(PGA) = poly(lambda)
    Inputs
      lambda_exceed : shape (n_rp,)
      ln_pga        : shape (n_rp, n_pts)
    Output
      coeffs        : shape (deg+1, n_pts) with NaNs where fit not possible
    """
    lam = np.asarray(lambda_exceed, dtype="float64")
    Y = np.asarray(ln_pga, dtype="float64")

    n_rp, n_pts = Y.shape
    coeffs = np.full((deg + 1, n_pts), np.nan, dtype="float64")

    # Vandermonde: columns correspond to polynomial terms
    V = np.vander(lam, deg + 1)  # shape (n_rp, deg+1)

    # only fit columns where all RP samples are finite
    good = np.all(np.isfinite(Y), axis=0) & np.all(np.isfinite(lam))
    if np.count_nonzero(good) < 1:
        return coeffs

    Yg = Y[:, good]  # shape (n_rp, n_good)
    # least squares in one shot: returns (deg+1, n_good)
    C, *_ = np.linalg.lstsq(V, Yg, rcond=None)
    coeffs[:, good] = C
    return coeffs


def process_one_area(area_id, areas_dir, rp_list, deg, out_dir, *,
                     pga_loaded=None, pga_tifs=None):
    """
    Read one area CSV, sample PGA rasters, fit ln(PGA) curve, save NPZ.
    Uses either:
      - pga_loaded (dict of loaded rasters)  [threading mode]
      - pga_tifs (dict rp->tif path)         [loky/process mode]
    """
    ar_csv = os.path.join(areas_dir, f"ar_{int(area_id):04d}.csv")
    if not os.path.exists(ar_csv):
        return dict(area_id=int(area_id), n_pts=0, out_npz="", status="missing_area_csv")

    df = pd.read_csv(ar_csv)
    if "x" not in df.columns or "y" not in df.columns:
        return dict(area_id=int(area_id), n_pts=0, out_npz="", status="missing_xy_columns")

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    n_pts = int(len(x))
    n_rp = int(len(rp_list))
    if n_pts < 1:
        return dict(area_id=int(area_id), n_pts=0, out_npz="", status="empty_area")

    rp_years = np.array([float(rp) for rp in rp_list], dtype="float64")
    lambda_exceed = 1.0 / rp_years

    pga = np.full((n_rp, n_pts), np.nan, dtype="float64")

    # sample each RP raster
    for i, rp in enumerate(rp_list):
        rps = str(rp)
        if pga_loaded is not None:
            # fast sampling from in-memory arrays
            pga[i, :] = sample_loaded_raster(pga_loaded[rps], x, y)
        else:
            # fallback (process mode): open & sample
            tif = pga_tifs[rps]
            with rasterio.open(tif) as src:
                pga[i, :] = sample_raster_points_open(src, x, y)

    ln_pga = np.log(np.clip(pga, 1e-12, np.inf))
    coeffs = fit_ln_pga_poly_vectorized(lambda_exceed, ln_pga, deg=deg)

    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"ar_{int(area_id):06d}_pga_fit.npz")

    np.savez_compressed(
        out_npz,
        area_id=int(area_id),
        x=np.rint(x).astype(np.int32),
        y=np.rint(y).astype(np.int32),
        rp_years=rp_years.astype(np.float32),
        lambda_exceed=lambda_exceed.astype(np.float32),
        deg=int(deg),
        ln_pga_coeffs=coeffs.astype(np.float64),
        pga_samples=pga.astype(np.float32),
    )

    return dict(area_id=int(area_id), n_pts=int(n_pts), out_npz=out_npz, status="ok")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 3 (areas): sample PGA rasters in each area and fit hazard curves.")
    ap.add_argument("--config", default="config.json", help="Path to case config.json")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel cores (-1 = all cores)")
    ap.add_argument("--backend", type=str, default="threading", choices=["threading", "loky"])
    ap.add_argument("--deg", type=int, default=2, help="Polynomial degree for ln(PGA) fit (default: 2).")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    if "outputs" not in cfg or "areas_dir" not in cfg["outputs"]:
        raise SystemExit("[FAIL] config missing: outputs.areas_dir")
    if "paths" not in cfg or "pga_rasters" not in cfg["paths"]:
        raise SystemExit("[FAIL] config missing: paths.pga_rasters")

    areas_dir = cfg["outputs"]["areas_dir"]
    index_csv = os.path.join(areas_dir, "index.csv")
    if not os.path.exists(index_csv):
        raise SystemExit(f"[FAIL] Missing index.csv: {index_csv}")

    idx = pd.read_csv(index_csv)
    if "id" not in idx.columns:
        raise SystemExit(f"[FAIL] index.csv missing 'id'. Columns: {list(idx.columns)}")
    area_ids = idx["id"].astype(int).tolist()

    pga_dict = cfg["paths"]["pga_rasters"]  # {"475": "...tif", ...}
    rp_list = sorted(pga_dict.keys(), key=float)
    if len(rp_list) < 2:
        raise SystemExit("[FAIL] Need at least 2 return periods to build a curve.")

    # clamp degree so it always makes sense
    deg = max(1, min(int(args.deg), len(rp_list) - 1))

    # ensure rasters exist
    pga_tifs = {}
    for rp in rp_list:
        tif = pga_dict[rp]
        if not os.path.exists(tif):
            raise SystemExit(f"[FAIL] Missing PGA raster for RP={rp}: {tif}")
        pga_tifs[str(rp)] = tif

    out_dir = os.path.join(areas_dir, "hazard_fits")
    os.makedirs(out_dir, exist_ok=True)

    # Pre-load rasters only for threading mode (shared memory; fast sampling)
    pga_loaded = None
    if args.backend == "threading":
        pga_loaded = {}
        for rp in rp_list:
            pga_loaded[str(rp)] = load_raster_band(pga_tifs[str(rp)], band=1, dtype="float32")

        # sanity: same shape/transform across PGA rasters (recommended)
        shapes = [pga_loaded[str(rp)]["shape"] for rp in rp_list]
        if any(s != shapes[0] for s in shapes):
            raise SystemExit("[FAIL] PGA rasters do not share the same shape. Align/clip them first.")

    prefer = "threads" if args.backend == "threading" else "processes"

    rows = Parallel(n_jobs=args.n_jobs, backend=args.backend, prefer=prefer, verbose=10)(
        delayed(process_one_area)(
            area_id=aid,
            areas_dir=areas_dir,
            rp_list=rp_list,
            deg=deg,
            out_dir=out_dir,
            pga_loaded=pga_loaded,
            pga_tifs=None if pga_loaded is not None else pga_tifs,
        )
        for aid in area_ids
    )

    out_index = os.path.join(out_dir, "hazard_fit_index.csv")
    pd.DataFrame(rows).to_csv(out_index, index=False)

    print(f"[OK] wrote {out_index}")
    print(f"[OK] degree used: {deg} (n_rp={len(rp_list)})")
    print(f"[OK] wrote NPZ files in: {out_dir}")


if __name__ == "__main__":
    main()