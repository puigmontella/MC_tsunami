#!/usr/bin/env python3
import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Point

IN_DIR = "transects"
CAND = os.path.join(IN_DIR, "candidate_segments.csv")
DEPTH_TIF = "../bati_32m_and100m_merged_EPSG32628.tif"

OUT_GPKG = os.path.join(IN_DIR, "candidate_segments.gpkg")
LAYER_SEG = "segments"
LAYER_PTS = "midpoints"
LAYER_TR  = "transects_all"   # optional but useful

cand = pd.read_csv(CAND)
if len(cand) == 0:
    raise RuntimeError("candidate_segments.csv is empty")

# CRS from raster
with rasterio.open(DEPTH_TIF) as src:
    crs = src.crs

seg_rows = []
pt_rows = []

# --- segments + midpoints (ALL candidates, ALL transects) ---
for r in cand.itertuples(index=False):
    tr_path = os.path.join(IN_DIR, r.csv)
    tr = pd.read_csv(tr_path)

    m = (tr["s_m"] >= r.s_start_m) & (tr["s_m"] <= r.s_end_m)
    sub = tr.loc[m, ["x", "y", "s_m", "depth", "slope_deg", "rough", "cls"]].copy()
    if len(sub) < 2:
        continue

    coords = list(zip(sub["x"].to_numpy(float), sub["y"].to_numpy(float)))
    geom = LineString(coords)
    mid = geom.interpolate(0.5, normalized=True)

    seg_rows.append({
        "transect_id": int(r.transect_id),
        "csv": str(r.csv),
        "s_start_m": float(r.s_start_m),
        "s_end_m": float(r.s_end_m),
        "len_m": float(r.len_m),
        "depth_mean": float(r.depth_mean),
        "depth_min": float(sub["depth"].min()),
        "depth_max": float(sub["depth"].max()),
        "slope_mean": float(r.slope_mean),
        "slope_max": float(sub["slope_deg"].max()),
        "rough_mean": float(r.rough_mean),
        "frac_sed": float(r.frac_sed),
        "geometry": geom
    })

    pt_rows.append({
        "transect_id": int(r.transect_id),
        "csv": str(r.csv),
        "len_m": float(r.len_m),
        "depth_mean": float(r.depth_mean),
        "slope_mean": float(r.slope_mean),
        "slope_max": float(sub["slope_deg"].max()),
        "rough_mean": float(r.rough_mean),
        "frac_sed": float(r.frac_sed),
        "geometry": Point(mid.x, mid.y)
    })

gdf_seg = gpd.GeoDataFrame(seg_rows, crs=crs)
gdf_pts = gpd.GeoDataFrame(pt_rows, crs=crs)

# --- full transects layer (optional but recommended) ---
tr_rows = []
for f in sorted(glob.glob(os.path.join(IN_DIR, "tr_*.csv"))):
    tr = pd.read_csv(f)
    if len(tr) < 2:
        continue
    geom = LineString(list(zip(tr["x"].to_numpy(float), tr["y"].to_numpy(float))))
    tr_id = int(tr["id"].iloc[0])
    tr_rows.append({
        "transect_id": tr_id,
        "csv": os.path.basename(f),
        "npts": int(len(tr)),
        "geometry": geom
    })

gdf_tr = gpd.GeoDataFrame(tr_rows, crs=crs)

# Write GeoPackage (overwrite)
if os.path.exists(OUT_GPKG):
    os.remove(OUT_GPKG)

gdf_seg.to_file(OUT_GPKG, layer=LAYER_SEG, driver="GPKG")
gdf_pts.to_file(OUT_GPKG, layer=LAYER_PTS, driver="GPKG")
gdf_tr.to_file(OUT_GPKG, layer=LAYER_TR, driver="GPKG")

print(f"[INFO] Wrote: {OUT_GPKG}")
print(f"  - {LAYER_SEG}: {len(gdf_seg)}")
print(f"  - {LAYER_PTS}: {len(gdf_pts)}")
print(f"  - {LAYER_TR}:  {len(gdf_tr)}")
