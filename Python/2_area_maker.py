#!/usr/bin/env python3
"""
2_area_maker.py

Build offshore "areas" as square tiles in raster CRS (UTM meters) and sample rasters
inside each tile (depth/slope/rough/class). Writes one CSV per area + an index.csv.

Inputs expected (from config.json)
----------------------------------
paths:
  bathy_tif
  coastline:
    shp   (coastline/contour geometry; will be reprojected to raster CRS if needed)

outputs:
  rasters_dir
  areas_dir

Also expects rasters from step 1 in outputs/rasters_dir:
  - slope_deg.tif
  - roughness_std.tif
  - substrate_proxy_class.tif

areas (REQUIRED; no defaults in this script for the core fields)
---------------------------------------------------------------
areas:
  tile_m          : tile size (meters) (square)
  stride_m        : spacing between tile origins (meters)
  sample_ds_m     : sampling step inside a tile (meters)
  depth_limit_m   : keep points with depth >= depth_limit_m (depth is negative offshore)
  min_valid_frac  : minimum fraction of valid offshore points required to keep the tile

Optional distance-to-shore filter (NEW)
---------------------------------------
areas:
  shore_dist_min_m : keep tiles with distance_to_shore >= this (meters). default 0
  shore_dist_max_m : keep tiles with distance_to_shore <= this (meters). default inf

Distance definition:
- We compute the distance (meters) from the TILE POLYGON (bbox) to the shoreline geometry.
  This is more robust than using only the tile center.

Run
---
python ../../Python/2_area_maker.py
python ../../Python/2_area_maker.py --plot-overview

Outputs (in outputs/areas_dir)
------------------------------
  - ar_0000.csv, ar_0001.csv, ...
  - index.csv

If --plot-overview:
  - areas_over_bathy.png
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import geopandas as gpd
from shapely.geometry import box

from utilities.geoio import (
    raster_pixel_size,
    load_raster_band,
    sample_loaded_raster,
    tile_bbox_from_indices,
)

# -------------------------
# Plotting
# -------------------------
def plot_areas_over_bathy(depth_tif, index_csv, out_png):
    idx = pd.read_csv(index_csv)

    with rasterio.open(depth_tif) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            data[data == nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    plt.figure(figsize=(10, 10))
    plt.imshow(np.ma.masked_invalid(data), extent=extent, origin="upper")
    plt.colorbar(label="bathy [m]", fraction=0.0415, pad=0.02)

    ax = plt.gca()

    cmap = plt.get_cmap("tab20")
    ncol = cmap.N

    for r in idx.itertuples(index=False):
        xmin, xmax, ymin, ymax = float(r.xmin), float(r.xmax), float(r.ymin), float(r.ymax)
        w = xmax - xmin
        h = ymax - ymin
        c = cmap(int(r.id) % ncol)

        ax.add_patch(
            Rectangle(
                (xmin, ymin), w, h,
                fill=False,
                linewidth=1.6,
                edgecolor=c,
            )
        )

        ax.text(
            float(r.x0), float(r.y0), str(int(r.id)),
            fontsize=8, color="k",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor=c,
                linewidth=1.0,
                alpha=0.85
            )
        )

    plt.title("Areas (tiles) over bathymetry")
    plt.xlabel("Easting [m] (UTM / raster CRS)")
    plt.ylabel("Northing [m] (UTM / raster CRS)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("[OK] wrote", out_png)


def _coast_unary_union(geoms):
    """Compatibility helper for shapely/geopandas versions."""
    try:
        # GeoSeries unary_union (may warn in newer versions, but works)
        return geoms.unary_union
    except Exception:
        # fallback
        from shapely.ops import unary_union
        return unary_union(list(geoms))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Build offshore tile areas and sample rasters inside them.")
    ap.add_argument("--config", default="config.json", help="Path to case config.json")
    ap.add_argument("--plot-overview", action="store_true", help="Write areas_over_bathy.png")
    args = ap.parse_args()

    # --- Load config ---
    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Required paths/outputs
    if "paths" not in cfg or "bathy_tif" not in cfg["paths"]:
        raise SystemExit("[FAIL] config missing: paths.bathy_tif")
    if "paths" not in cfg or "coastline" not in cfg["paths"] or "shp" not in cfg["paths"]["coastline"]:
        raise SystemExit("[FAIL] config missing: paths.coastline.shp")
    if "outputs" not in cfg or "rasters_dir" not in cfg["outputs"]:
        raise SystemExit("[FAIL] config missing: outputs.rasters_dir")
    if "outputs" not in cfg or "areas_dir" not in cfg["outputs"]:
        raise SystemExit("[FAIL] config missing: outputs.areas_dir")
    if "areas" not in cfg:
        raise SystemExit("[FAIL] config missing: areas block")

    depth_tif = cfg["paths"]["bathy_tif"]
    coast_shp = cfg["paths"]["coastline"]["shp"]
    rasters_dir = cfg["outputs"]["rasters_dir"]
    out_dir = cfg["outputs"]["areas_dir"]
    os.makedirs(out_dir, exist_ok=True)

    slope_tif = os.path.join(rasters_dir, "slope_deg.tif")
    rough_tif = os.path.join(rasters_dir, "roughness_std.tif")
    cls_tif   = os.path.join(rasters_dir, "substrate_proxy_class.tif")

    for p in [depth_tif, slope_tif, rough_tif, cls_tif, coast_shp]:
        if not os.path.exists(p):
            raise SystemExit(f"[FAIL] missing input: {p}")

    # --- REQUIRED area parameters (no defaults) ---
    acfg = cfg["areas"]
    required_keys = ["tile_m", "stride_m", "sample_ds_m", "depth_limit_m", "min_valid_frac"]
    for k in required_keys:
        if k not in acfg:
            raise SystemExit(f"[FAIL] config missing: areas.{k}")

    tile_m = float(acfg["tile_m"])
    stride_m = float(acfg["stride_m"])
    sample_ds_m = float(acfg["sample_ds_m"])
    depth_limit_m = float(acfg["depth_limit_m"])
    min_valid_frac = float(acfg["min_valid_frac"])

    if tile_m <= 0 or stride_m <= 0 or sample_ds_m <= 0:
        raise SystemExit("[FAIL] areas.tile_m, areas.stride_m, areas.sample_ds_m must be > 0")
    if not (0.0 < min_valid_frac <= 1.0):
        raise SystemExit("[FAIL] areas.min_valid_frac must be in (0,1]")

    # --- Optional distance-to-shore filter ---
    shore_dist_min_m = float(acfg.get("shore_dist_min_m", 0.0))
    shore_dist_max_m = float(acfg.get("shore_dist_max_m", np.inf))
    if shore_dist_min_m < 0:
        raise SystemExit("[FAIL] areas.shore_dist_min_m must be >= 0")
    if not (shore_dist_max_m > shore_dist_min_m):
        raise SystemExit("[FAIL] areas.shore_dist_max_m must be > areas.shore_dist_min_m")

    # --- Read raster reference info (CRS/transform/shape) ---
    with rasterio.open(depth_tif) as src_ref:
        transform = src_ref.transform
        crs = src_ref.crs
        height = src_ref.height
        width = src_ref.width

    if crs is None:
        raise SystemExit("[FAIL] bathy raster has no CRS")

    # --- Load coastline geometry in raster CRS ---
    coast = gpd.read_file(coast_shp)
    if coast.crs is None:
        raise SystemExit("[FAIL] coastline shapefile has no CRS defined")
    if coast.crs != crs:
        coast = coast.to_crs(crs)

    shore = _coast_unary_union(coast.geometry)
    # if coastline layer is polygons, use boundary as shoreline
    if shore.geom_type in ("Polygon", "MultiPolygon"):
        shore = shore.boundary

    # --- Load rasters into memory (fast sampling) ---
    depthR = load_raster_band(depth_tif, band=1, dtype="float32")
    slopeR = load_raster_band(slope_tif, band=1, dtype="float32")
    roughR = load_raster_band(rough_tif, band=1, dtype="float32")
    clsR   = load_raster_band(cls_tif, band=1, dtype="float32")

    # Sanity check: same shape
    if depthR["shape"] != slopeR["shape"] or depthR["shape"] != roughR["shape"] or depthR["shape"] != clsR["shape"]:
        raise SystemExit("[FAIL] rasters do not share the same shape; re-check preprocessing")

    dx, dy = raster_pixel_size(transform)

    # Convert meters -> pixels (separately for x/y)
    tile_ncols = int(round(tile_m / dx))
    tile_nrows = int(round(tile_m / dy))
    stride_ncols = int(round(stride_m / dx))
    stride_nrows = int(round(stride_m / dy))

    if tile_ncols < 1 or tile_nrows < 1:
        raise SystemExit("[FAIL] tile_m too small relative to pixel size")
    if stride_ncols < 1 or stride_nrows < 1:
        raise SystemExit("[FAIL] stride_m too small relative to pixel size")

    index_rows = []
    area_id = 0

    max_row0 = height - tile_nrows
    max_col0 = width - tile_ncols

    for row0 in range(0, max_row0 + 1, stride_nrows):
        for col0 in range(0, max_col0 + 1, stride_ncols):

            xmin, xmax, ymin, ymax = tile_bbox_from_indices(transform, row0, col0, tile_nrows, tile_ncols)

            # --- Shore-distance filter (tile polygon -> shore) ---
            tile_poly = box(xmin, ymin, xmax, ymax)
            dist_to_shore_m = float(tile_poly.distance(shore))  # meters (UTM)

            if (dist_to_shore_m < shore_dist_min_m) or (dist_to_shore_m > shore_dist_max_m):
                continue

            # Build sample points
            xs_1d = np.arange(xmin + 0.5 * sample_ds_m, xmax, sample_ds_m, dtype=float)
            ys_1d = np.arange(ymin + 0.5 * sample_ds_m, ymax, sample_ds_m, dtype=float)
            if xs_1d.size == 0 or ys_1d.size == 0:
                continue

            XX, YY = np.meshgrid(xs_1d, ys_1d)
            xs = XX.ravel()
            ys = YY.ravel()

            depth = sample_loaded_raster(depthR, xs, ys)
            slope = sample_loaded_raster(slopeR, xs, ys)
            rough = sample_loaded_raster(roughR, xs, ys)
            cls   = sample_loaded_raster(clsR, xs, ys)

            # Offshore + valid rules:
            # - depth finite
            # - depth < 0 (offshore)
            # - depth >= depth_limit_m (not deeper than limit; depth_limit_m is negative)
            m_valid = np.isfinite(depth) & (depth < 0.0) & (depth >= depth_limit_m)

            n_total = int(depth.size)
            n_valid = int(np.count_nonzero(m_valid))
            if n_total == 0:
                continue
            if (n_valid / float(n_total)) < min_valid_frac:
                continue
            if n_valid < 10:
                continue

            # Keep only valid points
            xs_k = xs[m_valid]
            ys_k = ys[m_valid]
            depth_k = depth[m_valid]
            slope_k = slope[m_valid]
            rough_k = rough[m_valid]
            cls_k   = cls[m_valid]

            # cls: NaN -> class 3, then int
            cls_k = np.where(np.isfinite(cls_k), cls_k, 3.0).astype("int32")

            # Area CSV
            df = pd.DataFrame({
                "id": int(area_id),
                "x": xs_k,
                "y": ys_k,
                "depth": depth_k,
                "slope_deg": slope_k,
                "rough": rough_k,
                "cls": cls_k,
            })

            # Precision policy: integers or round(3)
            df["x"] = df["x"].round(0).astype(int)
            df["y"] = df["y"].round(0).astype(int)
            df["depth"] = df["depth"].round(1)
            df["slope_deg"] = df["slope_deg"].round(3)
            df["rough"] = df["rough"].round(3)

            out_csv = os.path.join(out_dir, f"ar_{area_id:04d}.csv")
            df.to_csv(out_csv, index=False)

            # Index row
            x0 = int(round(0.5 * (xmin + xmax), 0))
            y0 = int(round(0.5 * (ymin + ymax), 0))

            depth_mean = float(np.nanmean(depth_k))
            slope_mean = float(np.nanmean(slope_k))
            rough_mean = float(np.nanmean(rough_k))

            index_rows.append({
                "id": int(area_id),
                "x0": int(x0),
                "y0": int(y0),
                "xmin": int(round(xmin, 0)),
                "xmax": int(round(xmax, 0)),
                "ymin": int(round(ymin, 0)),
                "ymax": int(round(ymax, 0)),
                "dist_to_shore_m": round(dist_to_shore_m, 1),
                "npoints": int(n_valid),
                "depth_mean": round(depth_mean, 3),
                "slope_mean": round(slope_mean, 3),
                "rough_mean": round(rough_mean, 3),
            })

            area_id += 1

    # Write index.csv
    index_path = os.path.join(out_dir, "index.csv")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)

    print(f"[OK] saved {len(index_rows)} areas in: {out_dir}")
    print("[OK] index:", index_path)
    print("[INFO] CRS:", crs)
    print(f"[INFO] shore distance filter: min={shore_dist_min_m} m  max={shore_dist_max_m} m")

    if args.plot_overview:
        out_png = os.path.join(out_dir, "areas_over_bathy.png")
        plot_areas_over_bathy(depth_tif, index_path, out_png)


if __name__ == "__main__":
    main()