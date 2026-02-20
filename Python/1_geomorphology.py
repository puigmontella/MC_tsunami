#!/usr/bin/env python3
import os, sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe on servers / no display
import matplotlib.pyplot as plt
import rasterio


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from utilities.config import load_config, abspath
from utilities.geoio import (
    read_raster1,
    write_raster1,
    raster_pixel_size,
    slope_degrees,
    highpass_roughness_std,
)

def main():
    ap = argparse.ArgumentParser(description="Compute slope, roughness, and substrate proxy from bathymetry.")
    ap.add_argument("--config", required=True, help="Path to case config.json")

    # optional overrides (if you want to test quickly without editing config)
    ap.add_argument("--out-dir", default=None, help="Override output rasters directory (else use config outputs/rasters_dir).")
    ap.add_argument("--rough-win-m", type=float, default=200.0,
                    help="Roughness window size in meters (converted to pixels).")
    ap.add_argument("--sigma-large-m", type=float, default=450.0,
                    help="Gaussian smoothing sigma for large-scale trend (meters).")
    ap.add_argument("--win-small-m", type=float, default=290.0,
                    help="Small window for texture std (meters).")
    ap.add_argument("--depth-cut", type=float, default=-2500.0,
                    help="Only classify substrate where bathy >= depth_cut (e.g. -2500).")
    ap.add_argument("--make-plots", action="store_true", help="Write diagnostic plots (PNG).")

    args = ap.parse_args()

    cfg = load_config(args.config)

    bathy_tif = abspath(cfg, cfg["paths"]["bathy_tif"])

    # Output directory
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = abspath(cfg, cfg["outputs"]["rasters_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # ---- read bathy
    bathy, prof, transform = read_raster1(bathy_tif)
    dx, dy = raster_pixel_size(transform)

    # convert meters -> pixels (keep odd windows)
    win_px = max(3, int(round(args.rough_win_m / dx)))
    if win_px % 2 == 0:
        win_px += 1

    sigma_large_px = max(1.0, float(args.sigma_large_m / dx))

    win_small_px = max(3, int(round(args.win_small_m / dx)))
    if win_small_px % 2 == 0:
        win_small_px += 1

    # ---- compute morphometrics
    slope = slope_degrees(bathy, dx, dy)
    rough = highpass_roughness_std(
        bathy,
        dx,
        sigma_large_px=sigma_large_px,
        win_small_px=win_small_px
    )

    out_slope = os.path.join(out_dir, "slope_deg.tif")
    out_rough = os.path.join(out_dir, "roughness_std.tif")
    write_raster1(out_slope, slope, prof, nodata=np.nan, dtype="float32")
    write_raster1(out_rough, rough, prof, nodata=np.nan, dtype="float32")

    # positive depth raster
    depth_pos = np.where(np.isfinite(bathy), -bathy, np.nan).astype("float32")
    out_depth = os.path.join(out_dir, "depth_pos.tif")
    write_raster1(out_depth, depth_pos, prof, nodata=np.nan, dtype="float32")

    # ---- substrate proxy classification
    band = np.isfinite(bathy) & (bathy >= float(args.depth_cut))

    if np.any(band):
        R95 = float(np.nanpercentile(rough[band], 95))
        R70 = float(np.nanpercentile(rough[band], 70))
        S95 = float(np.nanpercentile(slope[band], 95))
        S70 = float(np.nanpercentile(slope[band], 70))
    else:
        R95 = R70 = S95 = S70 = np.nan

    bedrock = (band & ((slope > S95) | ((slope > S70) & (rough > R95))))
    sediment = (band & ((slope < 25.0) & (rough < R70)))
    unknown = (band & (~bedrock) & (~sediment))

    # class codes:
    # 0 = sediment, 1 = unknown, 2 = bedrock, 3 = nodata
    cls = np.full(slope.shape, 1, dtype=np.uint8)
    cls[sediment] = 0
    cls[bedrock] = 2
    cls[~(np.isfinite(slope) & np.isfinite(rough) & band)] = 3

    out_cls = os.path.join(out_dir, "substrate_proxy_class.tif")
    out_prof = prof.copy()
    out_prof.update(dtype="uint8", count=1, nodata=3, compress="deflate")

    with rasterio.open(out_cls, "w", **out_prof) as dst:
        dst.write(cls, 1)

    n = int(band.sum())
    if n > 0:
        print("Sediment %:", 100.0 * float(sediment.sum()) / n)
        print("Bedrock %:", 100.0 * float(bedrock.sum()) / n)
        print("Unknown %:", 100.0 * float(unknown.sum()) / n)
    print("S70,S95:", S70, S95, " R70,R95:", R70, R95)

    print("[OK] wrote:", out_slope)
    print("[OK] wrote:", out_rough)
    print("[OK] wrote:", out_depth)
    print("[OK] wrote:", out_cls)

    # ---- optional plots
    if args.make_plots:
        rough_band = np.where(band, rough, np.nan)
        slope_band = np.where(band, slope, np.nan)

        plt.figure()
        plt.imshow(rough_band, interpolation="nearest", cmap="jet")
        plt.title("roughness (within band)")
        plt.colorbar()
        plt.clim(0, 25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roughness_band.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.imshow(slope_band, interpolation="nearest", cmap="viridis")
        plt.title("slope (within band)")
        plt.colorbar()
        plt.clim(0, 25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "slope_band.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.imshow(cls, interpolation="nearest", cmap="jet")
        plt.title("substrate_proxy_class (0 sed / 1 unk / 2 rock / 3 nodata)")
        plt.colorbar()
        plt.clim(0, 3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "substrate_proxy_class.png"), dpi=200)
        plt.close()

        print("[OK] wrote diagnostic PNGs in", out_dir)

if __name__ == "__main__":
    main()
