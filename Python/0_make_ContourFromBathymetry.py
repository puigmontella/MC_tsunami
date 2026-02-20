#!/usr/bin/env python3
import os
import argparse
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt


def affine_xy(transform, cols, rows):
    """Convert pixel coordinates (col,row) to map x,y using rasterio Affine."""
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    cols = np.asarray(cols, float)
    rows = np.asarray(rows, float)
    xs = a * cols + b * rows + c
    ys = d * cols + e * rows + f
    return xs, ys


def ensure_closed(ls):
    coords = list(ls.coords)
    if len(coords) < 4:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return LineString(coords)


def ensure_clockwise_closed_ring(ls):
    """If closed, enforce clockwise orientation (negative signed area)."""
    coords = list(ls.coords)
    if len(coords) < 4 or coords[0] != coords[-1]:
        return ls
    x = np.array([p[0] for p in coords], float)
    y = np.array([p[1] for p in coords], float)
    area2 = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])  # 2*signed area
    if area2 > 0:  # CCW -> reverse to CW
        coords = coords[::-1]
        return LineString(coords)
    return ls


def contour_to_lines(Zin, contour_level, transform, min_length_m):
    """
    Extract contour segments using cs.allsegs (matplotlib-version robust).
    Zin is a 2D array in pixel space.
    """
    # Make contour; do not rely on .collections/.get_paths()
    cs = plt.contour(Zin, levels=[contour_level])
    try:
        segs = cs.allsegs[0]  # list of (N,2) arrays in (x=col, y=row) coords
    finally:
        plt.close(cs.figure)

    out = []
    for seg in segs:
        if seg is None or len(seg) < 10:
            continue
        cols = seg[:, 0]
        rows = seg[:, 1]
        xs, ys = affine_xy(transform, cols, rows)
        ls = LineString(np.column_stack([xs, ys]))
        ls = ensure_closed(ls)
        if ls is None:
            continue
        if ls.length >= float(min_length_m):
            out.append(ls)
    return out


def extract_coastline(bathy_tif, level=0.0, method="auto", min_length_m=5000.0):
    with rasterio.open(bathy_tif) as src:
        Z = src.read(1).astype("float32")
        nod = src.nodata
        transform = src.transform
        crs = src.crs

    if nod is not None and not (isinstance(nod, float) and np.isnan(nod)):
        Z[Z == nod] = np.nan

    lines = []

    if method in ("auto", "contour0"):
        # If NaNs represent land/outside, set them to +1 so the 0-contour is the boundary
        Z0 = Z.copy()
        if np.isnan(Z0).any():
            Z0[np.isnan(Z0)] = float(level) + 1.0

        lines = contour_to_lines(Z0, float(level), transform, min_length_m)

        # If nothing found, fall back to mask boundary
        if method == "auto" and len(lines) == 0:
            method = "mask"

    if method == "mask":
        # Boundary between finite (sea grid) and NaN (land/outside)
        M = np.isfinite(Z).astype("float32")
        # contour at 0.5 gives the boundary of the valid mask
        lines = contour_to_lines(M, 0.5, transform, min_length_m)

    if len(lines) == 0:
        return None, crs

    merged = linemerge(lines)
    if merged.geom_type == "MultiLineString":
        merged = max(list(merged.geoms), key=lambda g: g.length)

    merged = ensure_closed(merged)
    if merged is None:
        return None, crs

    merged = ensure_clockwise_closed_ring(merged)
    return merged, crs


def main():
    ap = argparse.ArgumentParser(description="Generate a coastline polyline from bathymetry GeoTIFF.")
    ap.add_argument("--bathy-tif", required=True)
    ap.add_argument("--out-shp", required=True)
    ap.add_argument("--level", type=float, default=0.0, help="Coastline contour level (default 0 m).")
    ap.add_argument("--method", default="auto", choices=["auto", "contour0", "mask"],
                    help="auto: try 0-contour then mask; contour0: force 0-contour; mask: boundary of valid-data mask.")
    ap.add_argument("--min-length-m", type=float, default=5000.0, help="Drop tiny rings shorter than this.")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out_shp)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    line, crs = extract_coastline(args.bathy_tif, level=args.level, method=args.method, min_length_m=args.min_length_m)
    if line is None:
        raise SystemExit(
            "Could not extract coastline.\n"
            "Try:\n"
            "  --method contour0   (if bathy contains land elevations or NaN=land)\n"
            "  --method mask       (if NaN marks land/outside; bathy sea-only)\n"
        )

    gdf = gpd.GeoDataFrame({"id": [0]}, geometry=[line], crs=crs)
    gdf.to_file(args.out_shp)
    print("[OK] wrote", args.out_shp)


if __name__ == "__main__":
    main()
