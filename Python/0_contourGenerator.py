#!/usr/bin/env python3
"""
UNDER DEVELOPMENT 

0_contourGenerator.py

Generate a coastline-like contour from bathymetry and save it to:
  inputs/coastline/coast_from_bathy.shp

Run from cases/tenerife/:
  python ../../Python/0_contourGenerator.py

It reads bathy_tif from config.json, extracts the -20 m contour, smooths it,
and writes a shapefile with a single LineString feature.
"""

import os
import json
import argparse
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import LineString
from scipy.ndimage import gaussian_filter1d


def polyline_length_xy(xy):
    d = np.diff(xy, axis=0)
    return np.sum((d[:, 0]**2 + d[:, 1]**2)**0.5)


def resample_linestring(ls: LineString, ds: float) -> np.ndarray:
    """Resample LineString at constant spacing ds (meters in CRS units)."""
    L = float(ls.length)
    if L <= 0.0:
        return np.asarray(ls.coords, dtype=float)

    s = np.arange(0.0, L + ds, ds, dtype=float)
    pts = [ls.interpolate(float(si)) for si in s]
    return np.column_stack([[p.x for p in pts], [p.y for p in pts]]).astype(float)


def smooth_coords(xy: np.ndarray, smooth_m: float, ds: float) -> np.ndarray:
    """Gaussian smooth x(s), y(s) along polyline index."""
    if xy.shape[0] < 5 or smooth_m <= 0.0:
        return xy

    sigma_pts = max(1.0, float(smooth_m) / float(ds))
    x = gaussian_filter1d(xy[:, 0], sigma=sigma_pts, mode="nearest")
    y = gaussian_filter1d(xy[:, 1], sigma=sigma_pts, mode="nearest")
    return np.column_stack([x, y]).astype(float)


def main():
    ap = argparse.ArgumentParser(description="Generate coast_from_bathy.shp from bathymetry contour.")
    ap.add_argument("--config", default="config.json", help="Path to config.json (default: config.json)")
    ap.add_argument("--level", type=float, default=-20.0, help="Contour level in meters (default: -20)")
    ap.add_argument("--smooth-m", type=float, default=200.0, help="Smoothing length along the line (default: 200 m)")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    # Base directory = where config.json lives
    base_dir = os.path.dirname(os.path.abspath(args.config))

    with open(args.config, "r") as f:
        cfg = json.load(f)

    bathy_tif = cfg["paths"]["bathy_tif"]
    if not os.path.isabs(bathy_tif):
        bathy_tif = os.path.join(base_dir, bathy_tif)

    if not os.path.exists(bathy_tif):
        raise SystemExit(f"[FAIL] bathy_tif not found: {bathy_tif}")

    out_dir = base_dir + "/inputs/coastline"
    os.makedirs(out_dir, exist_ok=True)
    out_shp = out_dir + "/coast_from_bathy.shp"

    with rasterio.open(bathy_tif) as src:
        z = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            z[z == nodata] = np.nan

        # Build x,y axes (fast) for north-up rasters
        # If the raster is rotated/sheared, fallback to full grid.
        T = src.transform
        if (abs(T.b) < 1e-12) and (abs(T.d) < 1e-12):
            # x = c + (col + 0.5) * a
            # y = f + (row + 0.5) * e
            cols = np.arange(src.width, dtype=float)
            rows = np.arange(src.height, dtype=float)
            x = T.c + (cols + 0.5) * T.a
            y = T.f + (rows + 0.5) * T.e  # often negative step
            X, Y = x, y
            grid_mode = "1D"
        else:
            # Rare case: rotated rasters
            rr, cc = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing="ij")
            X, Y = rasterio.transform.xy(T, rr, cc, offset="center")
            X = np.asarray(X, dtype=float)
            Y = np.asarray(Y, dtype=float)
            grid_mode = "2D"

        # Extract contour segments using matplotlib
        fig = plt.figure()
        try:
            if grid_mode == "1D":
                cs = plt.contour(X, Y, np.ma.masked_invalid(z), levels=[args.level])
            else:
                cs = plt.contour(X, Y, np.ma.masked_invalid(z), levels=[args.level])

            segs = cs.allsegs[0] if cs.allsegs and len(cs.allsegs) > 0 else []
        finally:
            plt.close(fig)

        if not segs:
            raise SystemExit(f"[FAIL] No contour found at level {args.level}. Try another --level.")

        # Pick the longest segment
        segs_np = [np.asarray(s, dtype=float) for s in segs if len(s) >= 2]
        if not segs_np:
            raise SystemExit(f"[FAIL] Contour exists but no valid segments at level {args.level}.")

        lengths = [polyline_length_xy(s) for s in segs_np]
        coords = segs_np[int(np.argmax(lengths))]

        # Build line and smooth it (resample first)
        raw_line = LineString(coords)

        # Use pixel size as resampling step
        dx = float(abs(T.a)) if abs(T.a) > 0 else 10.0
        xy_rs = resample_linestring(raw_line, ds=dx)
        xy_sm = smooth_coords(xy_rs, smooth_m=float(args.smooth_m), ds=dx)
        smooth_line = LineString(xy_sm)

        # Write shapefile
        gdf = gpd.GeoDataFrame(
            {"level_m": [float(args.level)], "length_m": [float(smooth_line.length)]},
            geometry=[smooth_line],
            crs=src.crs,
        )
        gdf.to_file(out_shp, driver="ESRI Shapefile")

    print("[OK] wrote:", out_shp)


if __name__ == "__main__":
    main()