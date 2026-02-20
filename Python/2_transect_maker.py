#!/usr/bin/env python3
import os
import sys

import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.ops import linemerge
from shapely.geometry import Point, LineString, Polygon

from scipy.ndimage import gaussian_filter

# utilities import (project root)

# project root on PYTHONPATH (so imports work when running from anywhere)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utilities.config import load_config, abspath
from utilities.geoio import (
    read_raster1,
    write_raster1,
    raster_pixel_size,
    fill_nans_nearest,
)



# -------------------------
# Local helpers
# -------------------------
def unit_normals(line, s, eps=50.0):
    """Return (left_normal, right_normal) at curvilinear distance s along a LineString."""
    p0 = line.interpolate(max(0.0, s - eps))
    p1 = line.interpolate(min(line.length, s + eps))
    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    n = float(np.hypot(dx, dy))
    if n == 0.0:
        return None
    tx, ty = dx / n, dy / n
    return (-ty, tx), (ty, -tx)  # left, right

def sample_points(src, xs, ys):
    """Sample raster; convert nodata to NaN."""
    vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
    nd = src.nodata
    if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
        vals[vals == nd] = np.nan
    return vals

def compute_smoothed_slope_from_bathy(bathy_tif, out_slope_tif, smooth_m, overwrite=False):
    """
    Smooth bathy (Gaussian) then compute slope(deg) from gradients. Writes a slope raster.
    Uses fill_nans_nearest so smoothing doesn't spread NaNs.
    """
    if (not overwrite) and os.path.exists(out_slope_tif):
        print("[INFO] Using existing smoothed slope:", out_slope_tif)
        return out_slope_tif

    z, prof, transform = read_raster1(bathy_tif)
    dx, dy = raster_pixel_size(transform)

    zf = fill_nans_nearest(z)
    sigma_x = float(smooth_m / dx)
    sigma_y = float(smooth_m / dy)
    zs = gaussian_filter(zf, sigma=(sigma_y, sigma_x), mode="nearest")

    dzdy, dzdx = np.gradient(zs, dy, dx)  # (y,x)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad).astype("float32")
    slope_deg[np.isnan(z)] = np.nan

    write_raster1(out_slope_tif, slope_deg, prof, nodata=np.nan, dtype="float32")
    print("[INFO] Wrote smoothed slope ->", out_slope_tif)
    return out_slope_tif

def force_close_linestring(ls):
    coords = list(ls.coords)
    if len(coords) < 4:
        raise ValueError("Coastline polyline is too short/invalid.")
    if Point(coords[0]).distance(Point(coords[-1])) > 1e-6:
        coords.append(coords[0])
        ls = LineString(coords)
    return ls

def build_island_polygon(shore):
    poly = Polygon(shore).buffer(0)
    if poly.is_empty:
        raise ValueError("Failed to build island polygon from contour. Check contour geometry.")
    return poly

def plot_transects_over_bathy(depth_tif, index_csv, out_png, ds_m, title="Transects over bathymetry"):
    idx = pd.read_csv(index_csv)

    with rasterio.open(depth_tif) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            data[data == nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    plt.figure(figsize=(10, 10))
    plt.imshow(np.ma.masked_invalid(data), extent=extent, origin="upper")
    plt.colorbar(label="bathy (m)")

    for r in idx.itertuples(index=False):
        x0, y0 = float(r.x0), float(r.y0)
        nx, ny = float(r.nx), float(r.ny)
        L = (float(r.npoints) - 1.0) * float(ds_m)
        x1, y1 = x0 + nx * L, y0 + ny * L

        plt.plot([x0, x1], [y0, y1], linewidth=1)

        lab_off = 150.0
        xl = x0 + nx * lab_off
        yl = y0 + ny * lab_off
        plt.text(
            xl, yl, str(int(r.id)),
            fontsize=9, color="k",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7)
        )

    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("[OK] wrote", out_png)

def plot_profiles(transects_dir, transect_ids, out_dir, invert_depth=True):
    os.makedirs(out_dir, exist_ok=True)

    for tid in transect_ids:
        f = os.path.join(transects_dir, "tr_{:04d}.csv".format(int(tid)))
        if not os.path.exists(f):
            print("[SKIP] missing:", f)
            continue

        df = pd.read_csv(f)
        if df.empty:
            continue

        s = df["s_m"].to_numpy(float)
        depth = df["depth"].to_numpy(float)
        slope = df.get("slope_deg", pd.Series(np.nan, index=df.index)).to_numpy(float)
        rough = df.get("rough", pd.Series(np.nan, index=df.index)).to_numpy(float)

        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax[0].plot(s, depth, linewidth=1.5)
        ax[0].set_ylabel("depth (m)")
        if invert_depth:
            ax[0].invert_yaxis()
        ax[0].grid(True, alpha=0.3)
        ax[0].set_title("Transect {:04d}".format(int(tid)))

        ax[1].plot(s, slope, linewidth=1.5)
        ax[1].set_ylabel("slope (deg)")
        ax[1].grid(True, alpha=0.3)

        ax[2].plot(s, rough, linewidth=1.5)
        ax[2].set_ylabel("roughness")
        ax[2].set_xlabel("s along transect (m)")
        ax[2].grid(True, alpha=0.3)

        fig.tight_layout()

        out_png = os.path.join(out_dir, "profile_tr_{:04d}.png".format(int(tid)))
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
    ap.add_argument("--config", required=True, help="Path to case config.json")

    # Optional overrides (otherwise uses config)
    ap.add_argument("--coast-shp", default=None, help="Override coastline shapefile")
    ap.add_argument("--depth-tif", default=None, help="Override bathymetry tif")
    ap.add_argument("--out-dir", default=None, help="Override transects output directory")

    ap.add_argument("--make-smooth-slope", action="store_true",
                    help="Create a smoothed slope raster from bathy and use it (writes into rasters_dir).")
    ap.add_argument("--smooth-m", type=float, default=200.0, help="Smoothing length (meters) for smoothed slope.")
    ap.add_argument("--overwrite-smooth", action="store_true", help="Overwrite existing smoothed slope tif.")

    ap.add_argument("--plot-overview", action="store_true", help="Write transects_over_bathy.png")
    ap.add_argument("--plot-profiles", action="store_true", help="Write profile PNGs for selected ids.")
    ap.add_argument("--profile-ids", default="", help="Comma-separated transect IDs for profiles, e.g. '0,13,7,11'")
    ap.add_argument("--invert-depth", action="store_true", help="Invert depth axis in profiles (0 at top).")

    args = ap.parse_args()
    cfg = load_config(args.config)

    # ---- paths from config
    depth_tif = abspath(cfg, cfg["paths"]["bathy_tif"])
    rasters_dir = abspath(cfg, cfg["outputs"]["rasters_dir"])
    out_dir = abspath(cfg, cfg["outputs"]["transects_dir"])

    # coastline selection
    coast_cfg = cfg["paths"].get("coastline", {})
    coast_source = str(coast_cfg.get("source", "file")).lower()
    coast_shp = None
    if coast_source == "file":
        coast_shp = abspath(cfg, coast_cfg["shp"])
    else:
        # "generated" / "bathy" / etc => use generated_out
        coast_shp = abspath(cfg, coast_cfg["generated_out"])

    # apply optional overrides
    if args.depth_tif:
        depth_tif = args.depth_tif
    if args.coast_shp:
        coast_shp = args.coast_shp
    if args.out_dir:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # rasters produced by step 1
    slope_tif = os.path.join(rasters_dir, "slope_deg.tif")
    rough_tif = os.path.join(rasters_dir, "roughness_std.tif")
    cls_tif   = os.path.join(rasters_dir, "substrate_proxy_class.tif")

    # sanity
    for p in [depth_tif, coast_shp, slope_tif, rough_tif, cls_tif]:
        if not os.path.exists(p):
            raise SystemExit("[ERROR] Missing input: {}".format(p))

    # ---- transect params from config
    tcfg = cfg.get("transects", {})
    spacing_m     = float(tcfg.get("spacing_m", 8000.0))
    eps_m         = float(tcfg.get("eps_m", 50.0))
    max_len_m     = float(tcfg.get("max_len_m", 14000.0))
    ds_m          = float(tcfg.get("ds_m", 20.0))
    depth_limit_m = float(tcfg.get("depth_limit_m", -2500.0))

    # Optional: create smoothed slope and use it instead
    slope_tif_use = slope_tif
    if args.make_smooth_slope:
        slope_tif_smooth = os.path.join(rasters_dir, "slope_deg_smooth_{}m.tif".format(int(args.smooth_m)))
        compute_smoothed_slope_from_bathy(
            bathy_tif=depth_tif,
            out_slope_tif=slope_tif_smooth,
            smooth_m=float(args.smooth_m),
            overwrite=args.overwrite_smooth
        )
        slope_tif_use = slope_tif_smooth

    # ---- load coast + raster CRS
    coast = gpd.read_file(coast_shp)
    with rasterio.open(depth_tif) as tmp:
        rcrs = tmp.crs

    if coast.crs != rcrs:
        coast = coast.to_crs(rcrs)

    shore = linemerge(list(coast.geometry))
    if shore.geom_type == "MultiLineString":
        shore = max(list(shore.geoms), key=lambda g: g.length)

    shore = force_close_linestring(shore)
    island_poly = build_island_polygon(shore)

    # ---- open rasters once
    zsrc = rasterio.open(depth_tif)
    ssrc = rasterio.open(slope_tif_use)
    rsrc = rasterio.open(rough_tif)
    csrc = rasterio.open(cls_tif)

    index_rows = []
    transect_id = 0

    # ---- build & sample transects
    for coast_s in np.arange(0.0, shore.length, spacing_m):
        p = shore.interpolate(float(coast_s))

        normals = unit_normals(shore, float(coast_s), eps=eps_m)
        if normals is None:
            continue

        (nxL, nyL), _ = normals

        # Clockwise contour => sea is on LEFT
        nx, ny = float(nxL), float(nyL)

        # sanity flip: if it goes inside island, flip
        test_d = 200.0
        if island_poly.contains(Point(p.x + nx*test_d, p.y + ny*test_d)):
            nx, ny = -nx, -ny

        npts = int(max_len_m / ds_m) + 1
        s_m = np.arange(npts, dtype=float) * ds_m
        xs = p.x + nx * s_m
        ys = p.y + ny * s_m

        depth = sample_points(zsrc, xs, ys)
        slope = sample_points(ssrc, xs, ys)
        rough = sample_points(rsrc, xs, ys)
        cls   = sample_points(csrc, xs, ys)

        # keep contiguous segment until invalid OR deeper than limit
        keep_idx = []
        for i in range(npts):
            if not np.isfinite(depth[i]):
                break
            if depth[i] < depth_limit_m:
                break
            keep_idx.append(i)

        if len(keep_idx) < 10:
            continue

        keep_idx = np.array(keep_idx, dtype=int)

        df = pd.DataFrame({
            "id": transect_id,
            "s_m": s_m[keep_idx],
            "x": xs[keep_idx],
            "y": ys[keep_idx],
            "depth": depth[keep_idx],
            "slope_deg": slope[keep_idx],
            "rough": rough[keep_idx],
            "cls": cls[keep_idx].astype("int32"),
        })

        out_csv = os.path.join(out_dir, "tr_{:04d}.csv".format(transect_id))
        df.to_csv(out_csv, index=False)

        index_rows.append({
            "id": transect_id,
            "coast_s_m": float(coast_s),
            "x0": float(p.x),
            "y0": float(p.y),
            "nx": float(nx),
            "ny": float(ny),
            "npoints": int(len(keep_idx)),
            "csv": os.path.basename(out_csv)
        })

        transect_id += 1

    zsrc.close(); ssrc.close(); rsrc.close(); csrc.close()

    index_path = os.path.join(out_dir, "index.csv")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)

    print("[OK] Saved {} transects in: {}".format(len(index_rows), out_dir))
    print("[OK] index:", index_path)

    # overview plot
    if args.plot_overview:
        out_png = os.path.join(out_dir, "transects_over_bathy.png")
        plot_transects_over_bathy(depth_tif, index_path, out_png, ds_m)

    # profile plots
    if args.plot_profiles:
        if args.profile_ids.strip():
            tids = [int(x) for x in args.profile_ids.split(",") if x.strip() != ""]
        else:
            tids = list(range(min(10, len(index_rows))))

        prof_dir = os.path.join(out_dir, "profiles_png")
        plot_profiles(out_dir, tids, prof_dir, invert_depth=args.invert_depth)

if __name__ == "__main__":
    main()
