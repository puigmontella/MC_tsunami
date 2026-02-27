#!/usr/bin/env python3
"""
Compute slope, roughness, and a simple substrate proxy from bathymetry.

How to run
----------
From cases/tenerife/:

  python ../../Python/1_geomorphology.py

Optional examples:
  python ../../Python/1_geomorphology.py --make-plots
  python ../../Python/1_geomorphology.py --depth-cut -2500 --make-plots

Outputs (in outputs/rasters_dir by default)
-------------------------------------------
  - slope_deg.tif
  - roughness_std.tif
  - depth_pos.tif
  - substrate_proxy_class.tif

If --make-plots:
  - roughness_band.png
  - slope_band.png
  - substrate_proxy_class.png

 If you need more info run: 
 python ../../Python/1_geomorphology.py -h
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from utilities.geoio import (
    read_raster1,
    write_raster1,
    raster_pixel_size,
    slope_degrees,
    highpass_roughness_std,
    meters_to_pixels,
)


def main():
    ap = argparse.ArgumentParser(
        description="Compute slope, roughness, and substrate proxy from bathymetry."
    )
    # Default config so you don't need to pass --config every time
    ap.add_argument(
        "--config",
        default="config.json",
        help="Path to case config.json (default: config.json)",
    )

    ap.add_argument(
        "--sigma-large-m",
        type=float,
        default=450.0,
        help="Gaussian smoothing sigma for large scale trend in meters.",
    )
    ap.add_argument(
        "--win-small-m",
        type=float,
        default=290.0,
        help="Small window for texture std (meters)",
    )
    ap.add_argument(
        "--depth-cut",
        type=float,
        default=-3500.0,
        help="Only classify substrate where bathymetric values are >= depth_cut (for instance -3500).",
    )
    ap.add_argument(
        "--make-plots",
        action="store_true",
        help="Write plots in PNG format as well",
    )

    args = ap.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # -----------------------------
    # Paths 
    # -----------------------------
    bathy_tif = cfg["paths"]["bathy_tif"]
    print("[OK] bathy_tif:", bathy_tif)

    out_dir = cfg["outputs"]["rasters_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Read bathymetry
    # -----------------------------
    bathy, prof, transform = read_raster1(bathy_tif)
    dx, dy = raster_pixel_size(transform)  # meters per pixel (x/y)

    # -----------------------------
    # Convert "meters on the ground" -> "pixels in the raster"
    # -----------------------------
    sigma_large_px = max(1.0, float(args.sigma_large_m) / float(dx))
    win_small_px = meters_to_pixels(args.win_small_m, dx, force_odd=True)

    # -----------------------------
    # Morphometrics
    # -----------------------------
    slope = slope_degrees(bathy, dx, dy)

    rough = highpass_roughness_std(
        bathy,
        dx,
        sigma_large_px=sigma_large_px,
        win_small_px=win_small_px,
    )

    # -----------------------------
    # Write main rasters
    # -----------------------------
    out_slope = out_dir+"/slope_deg.tif"
    out_rough = out_dir+"/roughness_std.tif"

    write_raster1(out_slope, slope, prof, nodata=np.nan, dtype="float32")
    write_raster1(out_rough, rough, prof, nodata=np.nan, dtype="float32")

    depth_pos = np.where(np.isfinite(bathy), -bathy, np.nan).astype("float32")
    out_depth = out_dir+"/depth_pos.tif"
    write_raster1(out_depth, depth_pos, prof, nodata=np.nan, dtype="float32")

    # -----------------------------
    # Substrate proxy classification
    # -----------------------------
    band = np.isfinite(bathy) & (bathy >= float(args.depth_cut))

    R95 = float(np.nanpercentile(rough[band], 95))
    R70 = float(np.nanpercentile(rough[band], 70))
    S95 = float(np.nanpercentile(slope[band], 95))
    S70 = float(np.nanpercentile(slope[band], 70))


    bedrock = (band & ((slope > S95) | ((slope > S70) & (rough > R95))))
    sediment = (band & ((slope < 25.0) & (rough < R70)))
    unknown = (band & (~bedrock) & (~sediment))

    # class codes:
    #   0 = sediment
    #   1 = unknown
    #   2 = bedrock
    #   3 = nodata / outside band
    cls = np.full(slope.shape, 1, dtype=np.uint8)
    cls[sediment] = 0
    cls[bedrock] = 2
    cls[~(np.isfinite(slope) & np.isfinite(rough) & band)] = 3

    out_cls = out_dir+"/substrate_proxy_class.tif"
    out_prof = prof.copy()
    out_prof.update(dtype="uint8", count=1, nodata=3, compress="deflate")

    with rasterio.open(out_cls, "w", **out_prof) as dst:
        dst.write(cls, 1)

    # -----------------------------
    # Print short stats
    # -----------------------------
    n = int(band.sum())
    if n > 0:
        print("Sediment %:", round(100.0 * float(sediment.sum()) / n, 2))
        print("Bedrock %:", round(100.0 * float(bedrock.sum()) / n, 2))
        print("Unknown %:", round(100.0 * float(unknown.sum()) / n, 2))


    print("70th percentile of slope (deg):", round(S70, 2))
    print("95th percentile of slope (deg):", round(S95, 2))
    print("70th percentile of roughness:", round(R70, 2))
    print("95th percentile of roughness:", round(R95, 2))

    print("[OK] wrote:", out_slope)
    print("[OK] wrote:", out_rough)
    print("[OK] wrote:", out_depth)
    print("[OK] wrote:", out_cls)

    # -----------------------------
    # Optional plots
    # -----------------------------
    if args.make_plots:
        rough_band = np.where(band, rough, np.nan)
        plt.figure()
        plt.imshow(rough_band, interpolation="nearest", cmap="jet")
        plt.title("roughness (within band)")
        plt.colorbar()
        plt.clim(0, 25)
        plt.tight_layout()
        plt.savefig(out_dir+"/roughness_band.png", dpi=200)
        plt.close()

        slope_band = np.where(band, slope, np.nan)
        plt.figure()
        plt.imshow(slope_band, interpolation="nearest", cmap="viridis")
        plt.title("slope (within band)")
        plt.colorbar()
        plt.clim(0, 25)
        plt.tight_layout()
        plt.savefig(out_dir+"/slope_band.png", dpi=200)
        plt.close()

        plt.figure()
        plt.imshow(cls, interpolation="nearest", cmap="jet")
        plt.title("substrate_proxy_class (0 sed / 1 unk / 2 rock / 3 nodata)")
        plt.colorbar()
        plt.clim(0, 3)
        plt.tight_layout()
        plt.savefig(out_dir+"/substrate_proxy_class.png", dpi=200)
        plt.close()

        print("[OK] wrote diagnostic PNGs in", out_dir)


if __name__ == "__main__":
    main()