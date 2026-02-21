#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed

# Allow importing from project root (MC_tsunami/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utilities.geoio import load_raster_band, sample_loaded_raster
from utilities.config import load_config, abspath


# -----------------------------
# Sampling helpers
# -----------------------------
def sample_raster_at_points_chunked(src, xs, ys, chunk=200000):
    out = np.empty(len(xs), dtype=np.float32)
    out[:] = np.nan
    coords = np.column_stack([xs, ys])
    for i0 in range(0, len(coords), chunk):
        sl = slice(i0, min(len(coords), i0 + chunk))
        vals = np.array([v[0] for v in src.sample([tuple(xy) for xy in coords[sl]])], dtype=np.float32)
        out[sl] = vals
    return out.astype("float64")


def sample_pga(src_obj, xs, ys, chunk=200000):
    """src_obj can be RAM-loaded dict (load_raster_band) or rasterio dataset."""
    if isinstance(src_obj, dict):
        return sample_loaded_raster(src_obj, xs, ys)
    return sample_raster_at_points_chunked(src_obj, xs, ys, chunk=chunk)


def rp_sort_key(rp):
    try:
        return int(str(rp))
    except Exception:
        return 10**18


def _rp_years(rp):
    try:
        return float(str(rp))
    except Exception:
        return np.nan


def build_transect_points(idx_row, ds_m, s_max_m=None):
    """
    Generic transect point builder from index.csv row:
      x(s) = x0 + nx*s
      y(s) = y0 + ny*s
    Needs s_max_m (transect length). If your index.csv has L_m (or s_max_m), use it.
    """
    if s_max_m is None:
        # Try common column names
        for key in ["L_m", "length_m", "s_max_m", "max_s_m"]:
            if key in idx_row.index and np.isfinite(idx_row[key]):
                s_max_m = float(idx_row[key])
                break

    if s_max_m is None or not np.isfinite(s_max_m) or s_max_m <= 0:
        raise ValueError(
            "Missing transect length. Add L_m (or s_max_m) to index.csv or pass --s-max."
        )

    s = np.arange(0.0, s_max_m + 0.5 * ds_m, ds_m, dtype=float)
    x0, y0 = float(idx_row.x0), float(idx_row.y0)
    nx, ny = float(idx_row.nx), float(idx_row.ny)
    x = x0 + nx * s
    y = y0 + ny * s
    return s, x, y


def fit_lnpha_poly(prob_exceed, lnpha_matrix, deg):
    """
    Fit ln(PHA) = poly(prob_exceed) for ALL points at once.
    prob_exceed: shape (n_rp,)
    lnpha_matrix: shape (n_rp, n_pts)
    returns coeffs: shape (deg+1, n_pts) in np.polyval order (highest power first)
    """
    p = np.asarray(prob_exceed, dtype=float)
    Y = np.asarray(lnpha_matrix, dtype=float)

    # Mask invalid rows (some RP rasters might have NaNs everywhere at some points)
    # We'll fit point-by-point where possible.
    n_rp, n_pts = Y.shape
    coeffs = np.full((deg + 1, n_pts), np.nan, dtype=float)

    # Vandermonde for polynomial in p
    V = np.vander(p, deg + 1)  # (n_rp, deg+1)

    for j in range(n_pts):
        y = Y[:, j]
        m = np.isfinite(p) & np.isfinite(y)
        if np.sum(m) < (deg + 1):
            # not enough points to fit
            continue
        c, *_ = np.linalg.lstsq(V[m, :], y[m], rcond=None)
        coeffs[:, j] = c

    return coeffs


def process_one_transect(tid, idx_row, rp_list, pga_srcs, ds_m, s_max_override, k_factor, deg, chunk, out_dir):
    s, x, y = build_transect_points(idx_row, ds_m=ds_m, s_max_m=s_max_override)

    # Build (n_rp, n_pts) matrix of PHA in "g" units (dimensionless)
    rp_years = np.array([_rp_years(rp) for rp in rp_list], dtype=float)
    prob_exceed = 1.0 / rp_years  # annual probability of exceedance

    n_rp = len(rp_list)
    n_pts = len(s)
    pha_g = np.full((n_rp, n_pts), np.nan, dtype=float)

    for i, rp in enumerate(rp_list):
        src = pga_srcs[str(rp)]
        vals = sample_pga(src, x, y, chunk=chunk)

        # Convert to "g" if needed using k_factor.
        # If raster already in g, set k_factor=1.
        # If raster in m/s^2, set k_factor = 1/g.
        vals_g = vals * float(k_factor)
        pha_g[i, :] = vals_g

    # Fit polynomial ln(PHA_g) = poly(prob_exceed)
    # (Grilli: polynomial curve fits of ln(PHA) as a function of probability of exceedance) 
    lnpha = np.log(np.clip(pha_g, 1e-12, np.inf))
    coeffs = fit_lnpha_poly(prob_exceed, lnpha, deg=deg)

    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"tr_{int(tid):06d}_pha_fit.npz")
    np.savez_compressed(
        out_npz,
        transect_id=int(tid),
        s_m=s,
        x=x,
        y=y,
        rp_years=rp_years,
        prob_exceed=prob_exceed,
        deg=int(deg),
        lnpha_coeffs=coeffs,
        pha_g_samples=pha_g,  # keep samples for debugging/validation
    )

    return dict(transect_id=int(tid), n_pts=int(n_pts), out_npz=out_npz)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Step 3 (Grilli-style): build per-transect PHA hazard curve fits "
                    "(polynomial fits of ln(PHA) vs annual probability of exceedance)."
    )
    ap.add_argument("--config", required=True, help="Path to case config.json")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--backend", type=str, default="threading", choices=["threading", "loky", "multiprocessing"])
    ap.add_argument("--load-pga-to-ram", action="store_true")
    ap.add_argument("--no-load-pga-to-ram", action="store_true")
    ap.add_argument("--chunk", type=int, default=200000)

    ap.add_argument("--ds", type=float, default=None, help="Sampling step along transect (m). Default: config.pga_loading.ds_m or 250.")
    ap.add_argument("--s-max", type=float, default=None, help="Override transect length (m) if index.csv has no L_m/s_max_m.")
    ap.add_argument("--deg", type=int, default=None, help="Polynomial degree. Default: config.pga_loading.fit_degree or 3.")
    ap.add_argument("--k-factor", type=float, default=None,
                    help="Convert raster units to 'g'. If raster already in g -> 1. If raster in m/s^2 -> 1/9.81. "
                         "Default: config.pga_loading.k_factor or 1.")

    ap.add_argument("--rps", default=None, help="Comma list of RPs to use. Default: all config paths.pga_rasters keys.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"])
    index_csv = os.path.join(transects_dir, "index.csv")
    if not os.path.exists(index_csv):
        raise SystemExit(f"Missing index csv: {index_csv}")

    idx = pd.read_csv(index_csv).rename(columns={"id": "transect_id"})
    for c in ["transect_id", "x0", "y0", "nx", "ny"]:
        if c not in idx.columns:
            raise SystemExit(f"index.csv missing '{c}'. Columns: {list(idx.columns)}")

    pga_dict = cfg["paths"]["pga_rasters"]  # { "50": "pha_50yr.tif", ... }

    # RP list
    if args.rps:
        rp_list = [x.strip() for x in str(args.rps).split(",") if x.strip()]
    else:
        rp_list = sorted([str(k) for k in pga_dict.keys()], key=rp_sort_key)

    # params
    ds = float(args.ds) if args.ds is not None else float(cfg.get("pga_loading", {}).get("ds_m", 250.0))
    deg = int(args.deg) if args.deg is not None else int(cfg.get("pga_loading", {}).get("fit_degree", 3))
    k_factor_cfg = float(cfg.get("pga_loading", {}).get("k_factor", 1.0))
    k_factor = float(args.k_factor) if args.k_factor is not None else k_factor_cfg

    out_dir = abspath(cfg, cfg["outputs"].get("hazard_dir", os.path.join(transects_dir, "hazard_fits")))
    os.makedirs(out_dir, exist_ok=True)

    # RAM loading policy
    load_to_ram = True
    if args.no_load_pga_to_ram:
        load_to_ram = False
    if args.load_pga_to_ram:
        load_to_ram = True

    # Load rasters
    pga_srcs = {}
    for rp in rp_list:
        if rp not in pga_dict:
            raise SystemExit(f"RP {rp} requested but not in config paths.pga_rasters")
        tif = abspath(cfg, pga_dict[rp])
        if not os.path.exists(tif):
            raise SystemExit(f"Missing PGA/PHA raster for RP={rp}: {tif}")

        if load_to_ram:
            pga_srcs[rp] = load_raster_band(tif, band=1, dtype="float32")
        else:
            pga_srcs[rp] = rasterio.open(tif)

    tids = idx["transect_id"].astype(int).tolist()

    prefer = "threads" if args.backend == "threading" else "processes"
    rows = Parallel(n_jobs=args.n_jobs, backend=args.backend, prefer=prefer, verbose=10)(
        delayed(process_one_transect)(
            tid=int(tid),
            idx_row=idx[idx["transect_id"] == int(tid)].iloc[0],
            rp_list=rp_list,
            pga_srcs=pga_srcs,
            ds_m=ds,
            s_max_override=args.s_max,
            k_factor=k_factor,
            deg=deg,
            chunk=int(args.chunk),
            out_dir=out_dir,
        )
        for tid in tids
    )

    out_index = os.path.join(out_dir, "hazard_fit_index.csv")
    pd.DataFrame(rows).to_csv(out_index, index=False)
    print(f"[OK] wrote {out_index}")

    # Close rasters if needed
    if not load_to_ram:
        for src in pga_srcs.values():
            try:
                src.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()