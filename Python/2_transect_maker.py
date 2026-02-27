#!/usr/bin/env python3
"""
Build offshore transects from a coastline and sample rasters along them (depth/slope/rough/class).

How to run
----------
From cases/tenerife/:

  python ../../Python/2_transect_maker.py

Optional examples:
  python ../../Python/2_transect_maker.py --plot-overview
  python ../../Python/2_transect_maker.py --plot-profiles --profile-ids 0,5,12 
  python ../../Python/2_transect_maker.py --make-smooth-slope --smooth-m 200 --plot-overview

Inputs expected (relative paths in config.json)
----------------------------------------------
  paths:
    bathy_tif
    coastline: {source: "file"|"generated", shp|generated_out}
  outputs:
    rasters_dir
    transects_dir

Also expects rasters from step 1 in outputs/rasters_dir:
  - slope_deg.tif
  - roughness_std.tif
  - substrate_proxy_class.tif

Outputs (in outputs/transects_dir)
----------------------------------
  - tr_0000.csv, tr_0001.csv, ...
  - index.csv

If --plot-overview:
  - transects_over_bathy.png

If --plot-profiles:
  - profiles_png/profile_tr_XXXX.png

Help
----
python ../../Python/2_transect_maker.py -h

"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString
from scipy.ndimage import gaussian_filter

from utilities.geoio import (
    read_raster1,
    write_raster1,
    raster_pixel_size,
    fill_nans_nearest,
    sample_raster_points,
)

# -------------------------
# Geometry functions
# -------------------------
def unit_normals(line, s, eps):
    """
    Return normals (both directions) at curvilinear distance s along a line.
    eps controls how far we look forward/backward along the line.
    """
    p0 = line.interpolate(max(0.0, s - eps)) #1st point at distance s-eps 
    p1 = line.interpolate(min(line.length, s + eps))  #2nd point at distance s+eps
    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    n=(dx**2 + dy**2)**0.5
    if n == 0.0:
        return None
    tx, ty = dx / n, dy / n
    return (-ty, tx), (ty, -tx)  # left and right normals 


def mean_depth_along_normal(point, n, zsrc, test_ds=(100.0, 300.0, 800.0)):
    """
    Sample bathymetry a few meters away from the coast along a candidate normal direction.
    Return the mean depth (more negative => deeper).
    """
    xs = [point.x + n[0] * d for d in test_ds]
    ys = [point.y + n[1] * d for d in test_ds]
    zz = np.asarray(sample_raster_points(zsrc, xs=xs, ys=ys), dtype=float)
    zz = zz[np.isfinite(zz)] #take only the valid samples (ignore nodata/NaN)
    return np.mean(zz) if zz.size else float("nan")


def offshore_normal_from_bathy(point, nL, nR, zsrc, test_ds=(100.0, 300.0, 800.0)):
    """
    Offshore normal set on the side that becomes deeper
    """
    zL = mean_depth_along_normal(point, nL, zsrc, test_ds)
    zR = mean_depth_along_normal(point, nR, zsrc, test_ds)

    # If only one side is valid, use it
    if np.isfinite(zL) and np.isnan(zR):
        return nL[0], nL[1]
    if np.isfinite(zR) and np.isnan(zL):
        return nR[0], nR[1]

    # If both valid, pick the deeper side (more negative)
    if np.isfinite(zL) and np.isfinite(zR):
        return (nL[0], nL[1]) if (zL < zR) else (nR[0], nR[1])

    # keep left normal
    return nL[0], nL[1]


# -------------------------
# Raster smoothing
# -------------------------
def compute_smoothed_slope_from_bathy(bathy_tif, out_slope_tif, smooth_m):
    """
    Make an alternative slope raster more smooth.
    This is useful if raw bathy is noisy and creates unrealistically spiky slopes.
    """
    z, prof, transform = read_raster1(bathy_tif)
    dx, dy = raster_pixel_size(transform)

    zf = fill_nans_nearest(z)
    sigma_x = float(smooth_m / dx)
    sigma_y = float(smooth_m / dy)
    zs = gaussian_filter(zf, sigma=(sigma_y, sigma_x), mode="nearest")

    dzdy, dzdx = np.gradient(zs, dy, dx)  # (y,x)
    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype("float32")
    slope_deg[np.isnan(z)] = np.nan  # keep original nodata

    write_raster1(out_slope_tif, slope_deg, prof, nodata=np.nan, dtype="float32")
    print("[OK] wrote smoothed slope ->", out_slope_tif)
    return out_slope_tif


# -------------------------
# Plotting
# -------------------------
def plot_transects_over_bathy(depth_tif, index_csv, out_png, ds_m):
    """
    Quick map overview of transects over the bathymetry raster:
    """
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

    for r in idx.itertuples(index=False):
        # starting point and transect direction
        x0, y0 = r.x0, r.y0
        nx, ny = r.nx, r.ny

        # physical length based on number of points and ds
        L = (r.npoints - 1.0) * ds_m
        x1, y1 = x0 + nx * L, y0 + ny * L

        plt.plot([x0, x1], [y0, y1], linewidth=1)

        # text label slightly offshore
        lab_off = 150.0
        xl = x0 + nx * lab_off
        yl = y0 + ny * lab_off
        plt.text(
            xl, yl, str(int(r.id)),
            fontsize=9, color="k",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7)
        )

    plt.title("Transects over bathymetry")
    plt.xlabel("Longitude [m]")
    plt.ylabel("Latitude [m]")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("[OK] wrote", out_png)


def plot_profiles(transects_dir, transect_ids, out_dir):
    """
    For the chosen transects, plot 3 curves vs distance s:
      - depth(s)
      - slope(s)
      - roughness(s)
    """
    os.makedirs(out_dir, exist_ok=True)

    for tid in transect_ids:
        f = transects_dir + f"/tr_{int(tid):04d}.csv"
        if not os.path.exists(f):
            print("[SKIP] missing:", f)
            continue

        df = pd.read_csv(f)

        s = df["s_m"].to_numpy(float)
        depth = df["depth"].to_numpy(float)
        slope = pd.to_numeric(df.get("slope_deg", np.nan)).to_numpy(float)
        rough = pd.to_numeric(df.get("rough", np.nan)).to_numpy(float)

        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax[0].plot(s, depth, linewidth=1.5)
        ax[0].set_ylabel("depth [m]")
        ax[0].grid(True, alpha=0.3)
        ax[0].set_title(f"Transect {int(tid):04d}")

        ax[1].plot(s, slope, linewidth=1.5)
        ax[1].set_ylabel("slope [deg]")
        ax[1].grid(True, alpha=0.3)

        ax[2].plot(s, rough, linewidth=1.5)
        ax[2].set_ylabel("roughness")
        ax[2].set_xlabel("s along transect [m]")
        ax[2].grid(True, alpha=0.3)

        fig.tight_layout()

        out_png = out_dir + f"/profile_tr_{int(tid):04d}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print("[OK]", out_png)

    print("[DONE] profile plots in", out_dir)

    

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build offshore transects from coastline and sample rasters (depth/slope/rough/class)."
    )

    ap.add_argument("--config", default="config.json",
                    help="Path to case config.json (default: config.json)")

    # Optional
    ap.add_argument("--make-smooth-slope", action="store_true",
                    help="Create a smoothed slope raster from bathy and use it.")
    ap.add_argument("--smooth-m", type=float, default=200.0,
                    help="Smoothing length (in meters) for smoothed slope.")

    ap.add_argument("--plot-overview", action="store_true", help="Write transects_over_bathy.png")

    ap.add_argument("--plot-profiles", action="store_true", help="Write profile PNGs for selected IDs.")
    ap.add_argument("--profile-ids", default="", help="Comma-separated transect IDs for profiles, e.g. '0,13,7,11'")

    args = ap.parse_args()

    # Read case config
    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # paths from config
    depth_tif = cfg["paths"]["bathy_tif"]
    rasters_dir = cfg["outputs"]["rasters_dir"]
    out_dir = cfg["outputs"]["transects_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # This grabs the coastline block from the config.json
    coast_cfg = cfg["paths"]["coastline"]
    coast_shp = coast_cfg["shp"]

    # reading rasters produced by step 1
    slope_tif = rasters_dir + "/slope_deg.tif"
    rough_tif = rasters_dir + "/roughness_std.tif"
    cls_tif   = rasters_dir + "/substrate_proxy_class.tif"

    # sanity check inputs
    for p in [depth_tif, coast_shp, slope_tif, rough_tif, cls_tif]:
        if not os.path.exists(p):
            raise SystemExit(f"[FAIL] missing input: {p}")

    # ---- transect params from config -----
    # spacing_m: distance along coastline between transect origins
    # eps_m: used to estimate local tangent/normal
    # max_len_m: maximum offshore length of a transect
    # ds_m: sampling step along transect (points every ds_m meters)
    # depth_limit_m: stop the transect once bathy goes deeper than this threshold
    tcfg = cfg.get("transects", {})
    spacing_m     = float(tcfg.get("spacing_m", 8000.0))
    eps_m         = float(tcfg.get("eps_m", 50.0))
    max_len_m     = float(tcfg.get("max_len_m", 14000.0))
    ds_m          = float(tcfg.get("ds_m", 20.0))
    depth_limit_m = float(tcfg.get("depth_limit_m", -2500.0))

    # smoothing the bathymetry if requested, and use the smoothed slope instead of the original one
    slope_tif_use = slope_tif
    if args.make_smooth_slope:
        slope_tif_smooth = rasters_dir+ f"/slope_deg_smooth_{int(args.smooth_m)}m.tif"
        slope_tif_use = compute_smoothed_slope_from_bathy(
            bathy_tif=depth_tif,
            out_slope_tif=slope_tif_smooth,
            smooth_m=float(args.smooth_m),
        )

    # ---- load coast + raster CRS
    # coastline geometry must be in the same CRS as the rasters (same coordinate units)
    coast = gpd.read_file(coast_shp)
    with rasterio.open(depth_tif) as tmp:
        rcrs = tmp.crs

    # check CRS and reproject coast if needed
    if coast.crs != rcrs:
        coast = coast.to_crs(rcrs)

    shore = coast.geometry.iloc[0]


    # ---- open rasters -----
    zsrc = rasterio.open(depth_tif)
    ssrc = rasterio.open(slope_tif_use)
    rsrc = rasterio.open(rough_tif)
    csrc = rasterio.open(cls_tif)

    index_rows = []
    transect_id = 0

    # ---- build and  sample transects
    # coast_s is curvilinear distance along the coastline.
    # We place a new transect every spacing_m meters along the shore.
    for coast_s in np.arange(0.0, shore.length, spacing_m):
        # point on the coastline where this transect starts
        p = shore.interpolate(coast_s)

        # local normals at that coastline position
        normals = unit_normals(shore, coast_s, eps=eps_m)

        # take BOTH normals
        (nL, nR) = normals

        # choose the offshore direction by checking which side becomes deeper
        nx, ny = offshore_normal_from_bathy(p, nL, nR, zsrc)

        # build transect points along the chosen normal direction
        npts = int(max_len_m / ds_m) + 1
        s_m = np.arange(npts, dtype=float) * ds_m
        xs = p.x + nx * s_m
        ys = p.y + ny * s_m

        # sample each raster at all transect points
        depth = sample_raster_points(zsrc, xs=xs, ys=ys)
        slope = sample_raster_points(ssrc, xs=xs, ys=ys)
        rough = sample_raster_points(rsrc, xs=xs, ys=ys)
        cls   = sample_raster_points(csrc, xs=xs, ys=ys)

        # keep a contiguous offshore segment until:
        #   - we leave the raster / hit nodata, OR
        #   - we cross the depth limit (depth_limit_m)
        keep = []
        for i in range(npts):
            if not np.isfinite(depth[i]):
                break
            if depth[i] < depth_limit_m:
                break
            keep.append(i)


        keep = np.asarray(keep, dtype=int)

        # keep cls (class of substrate) as int. If NaN -> map them to class "3"
        cls_keep = cls[keep]
        cls_keep = np.where(np.isfinite(cls_keep), cls_keep, 3.0).astype("int32")

        # transect CSV: one row per sample point along the transect
        df = pd.DataFrame({
            "id": transect_id,
            "s_m": s_m[keep],
            "x": xs[keep],
            "y": ys[keep],
            "depth": depth[keep],
            "slope_deg": slope[keep],
            "rough": rough[keep],
            "cls": cls_keep,
        })

        # choose your precision here
        df["s_m"]       = df["s_m"].round(0)      # e.g. 0.1 m
        df["depth"]     = df["depth"].round(1)    # e.g. cm
        df["slope_deg"] = df["slope_deg"].round(3)
        df["rough"]     = df["rough"].round(3)

        # if x,y are in meters and you want integers:
        df["x"] = df["x"].round(0).astype(int)
        df["y"] = df["y"].round(0).astype(int)

        out_csv = out_dir + f"/tr_{transect_id:04d}.csv"
        df.to_csv(out_csv, index=False)

            # index.csv: one row per transect
        index_rows.append({
            "id": transect_id,
            "coast_s_m": round(float(coast_s), 0),
            "x0": round(float(p.x), 0),
            "y0": round(float(p.y), 0),
            "nx": round(float(nx), 4),
            "ny": round(float(ny), 4),
            "npoints": int(np.count_nonzero(keep)),
            "csv": os.path.basename(out_csv),
        })

        transect_id += 1

    # close raster handles
    zsrc.close()
    ssrc.close()
    rsrc.close()
    csrc.close()

    # write index file
    index_path = out_dir + "/index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)

    print(f"[OK] saved {len(index_rows)} transects in: {out_dir}")
    print("[OK] index:", index_path)

    # overview plot (map view)
    if args.plot_overview:
        out_png = out_dir + "/transects_over_bathy.png"
        plot_transects_over_bathy(depth_tif, index_path, out_png, ds_m)

    # profile plots (depth/slope/rough vs distance)
    if args.plot_profiles:
        if args.profile_ids.strip():
            tids = [int(x) for x in args.profile_ids.split(",") if x.strip() != ""]
        else:
            raise SystemExit("Please provide transect IDs for profiles using --profile-ids, e.g. --profile-ids 0,5,12")

        prof_dir = out_dir + "/profiles_png"
        plot_profiles(out_dir, tids, prof_dir)


if __name__ == "__main__":
    main()