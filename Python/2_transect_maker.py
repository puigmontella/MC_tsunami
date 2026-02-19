#!/usr/bin/env python3
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.fill import fillnodata
from shapely.ops import linemerge
from shapely.geometry import Point, LineString, Polygon
import os

# for smoothing
from scipy.ndimage import gaussian_filter

# -------------------------
# INPUTS (edit)
# -------------------------
DEPTH_TIF = "../../MC_tsunami_out/bati_32m_and100m_merged_EPSG32628.tif"
COAST_VEC = "../contourTenerife_v2.shp"   # almost-closed polyline around Tenerife (CLOCKWISE)

# Raw rasters
SLOPE_TIF_RAW = "slope_deg.tif"
ROUGH_TIF = "roughness_std.tif"
CLS_TIF   = "substrate_proxy_class.tif"   # 0 sed, 1 rock, 2 unk, nodata=3 

plotFlag = True

OUT_DIR = "transects"
os.makedirs(OUT_DIR, exist_ok=True)

# Transect layout
SPACING_M = 8000
EPS_M     = 50
MAX_LEN_M = 14000

# Sampling along transect
DS_M = 20.0
DEPTH_LIMIT = -2500

# -------------------------
# Slope smoothing controls
# -------------------------
MAKE_SMOOTH_SLOPE = True
SMOOTH_M = 200.0  # smoothing length in meters (try 80–150 m)
SLOPE_TIF_SMOOTH = f"slope_deg_smooth_{int(SMOOTH_M)}m.tif"
OVERWRITE_SMOOTH = True

# -------------------------
# Helpers
# -------------------------
def unit_normals(line, s, eps=50.0):
    """Return (left_normal, right_normal) at curvilinear distance s along a LineString."""
    p0 = line.interpolate(max(0.0, s - eps))
    p1 = line.interpolate(min(line.length, s + eps))
    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    n = (dx*dx + dy*dy) ** 0.5
    if n == 0:
        return None
    tx, ty = dx/n, dy/n
    return (-ty, tx), (ty, -tx)  # left, right

def sample_points(src, xs, ys):
    """Sample raster; convert nodata to NaN."""
    vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
    nd = src.nodata
    if nd is not None:
        if isinstance(nd, float) and np.isnan(nd):
            # already NaN nodata
            pass
        else:
            vals[vals == nd] = np.nan
    return vals

def ensure_smoothed_slope(depth_tif, out_slope_tif, smooth_m=100.0, overwrite=False):
    """Smooth bathy then compute slope(deg) and save."""
    if (not overwrite) and os.path.exists(out_slope_tif):
        print(f"[INFO] Using existing smoothed slope: {out_slope_tif}")
        return

    with rasterio.open(depth_tif) as src:
        z = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nod = src.nodata

        # mask nodata
        if nod is not None:
            z_mask = (z == nod)
        else:
            z_mask = ~np.isfinite(z)

        z = z.astype(np.float32)
        z[z_mask] = np.nan

        dx = abs(src.transform.a)
        dy = abs(src.transform.e)

    # Fill nodata (so smoothing doesn’t spread NaNs)
    # fillnodata expects a mask where 1 indicates "to fill"
    z_filled = z.copy()
    if np.isnan(z_filled).any():
        z_filled = fillnodata(z_filled, mask=np.isnan(z_filled), max_search_distance=50, smoothing_iterations=0)

        # if still NaNs (big gaps), fall back to median
        if np.isnan(z_filled).any():
            z_filled[np.isnan(z_filled)] = np.nanmedian(z_filled)

    # Gaussian smoothing in pixel units
    sigma_x = smooth_m / dx
    sigma_y = smooth_m / dy
    z_s = gaussian_filter(z_filled, sigma=(sigma_y, sigma_x), mode="nearest")

    # Slope from gradients
    dzdy, dzdx = np.gradient(z_s, dy, dx)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Put NaN back where original was nodata
    slope_deg[z_mask] = np.nan

    # Save
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(out_slope_tif, "w", **profile) as dst:
        dst.write(slope_deg, 1)

    print(f"[INFO] Wrote smoothed slope -> {out_slope_tif}")

# -------------------------
# Make smoothed slope and choose which slope raster to use
# -------------------------
if MAKE_SMOOTH_SLOPE:
    ensure_smoothed_slope(DEPTH_TIF, SLOPE_TIF_SMOOTH, smooth_m=SMOOTH_M, overwrite=OVERWRITE_SMOOTH)
    SLOPE_TIF = SLOPE_TIF_SMOOTH
else:
    SLOPE_TIF = SLOPE_TIF_RAW

# -------------------------
# Load coast + rasters
# -------------------------
coast = gpd.read_file(COAST_VEC)

with rasterio.open(DEPTH_TIF) as tmp:
    rcrs = tmp.crs

# reproject coast to raster CRS if needed
if coast.crs != rcrs:
    coast = coast.to_crs(rcrs)

# merge linework -> pick the longest line around the island
shore = linemerge(list(coast.geometry))
if shore.geom_type == "MultiLineString":
    shore = max(list(shore.geoms), key=lambda g: g.length)

# ---- force-close (your line is "almost closed") ----
coords = list(shore.coords)
if len(coords) < 4:
    raise ValueError("Coastline/contour line is too short or invalid.")

gap = Point(coords[0]).distance(Point(coords[-1]))
if gap > 1e-6:
    coords.append(coords[0])
    shore = LineString(coords)

# polygon used only for sanity flipping
island_poly = Polygon(shore).buffer(0)
if island_poly.is_empty:
    raise ValueError("Failed to build island polygon from contour. Check contour geometry.")

# open rasters once
zsrc = rasterio.open(DEPTH_TIF)
ssrc = rasterio.open(SLOPE_TIF)
rsrc = rasterio.open(ROUGH_TIF)
csrc = rasterio.open(CLS_TIF)

index_rows = []
transect_id = 0

# -------------------------
# Build & sample transects
# -------------------------
for s0 in np.arange(0, shore.length, SPACING_M):
    p = shore.interpolate(s0)

    normals = unit_normals(shore, s0, eps=EPS_M)
    if normals is None:
        continue
    (nxL, nyL), (nxR, nyR) = normals

    # contour CLOCKWISE => sea is on LEFT side
    nx, ny = nxL, nyL

    # sanity flip: if direction goes inside island, flip it
    test_d = 200.0
    if island_poly.contains(Point(p.x + nx*test_d, p.y + ny*test_d)):
        nx, ny = -nx, -ny

    # sample points along transect
    npts = int(MAX_LEN_M / DS_M) + 1
    s_m = np.arange(npts, dtype=float) * DS_M
    xs = p.x + nx * s_m
    ys = p.y + ny * s_m

    depth = sample_points(zsrc, xs, ys)
    slope = sample_points(ssrc, xs, ys)   # now from smoothed slope raster
    rough = sample_points(rsrc, xs, ys)
    cls   = sample_points(csrc, xs, ys)

    # Keep contiguous segment from coast until invalid OR deeper than DEPTH_LIMIT
    keep_idx = []
    for i in range(npts):
        if not np.isfinite(depth[i]):
            break
        if depth[i] < DEPTH_LIMIT:
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

    out_csv = os.path.join(OUT_DIR, f"tr_{transect_id:04d}.csv")
    df.to_csv(out_csv, index=False)

    index_rows.append({
        "id": transect_id,
        "coast_s_m": float(s0),
        "x0": float(p.x),
        "y0": float(p.y),
        "nx": float(nx),
        "ny": float(ny),
        "npoints": int(len(keep_idx)),
        "csv": os.path.basename(out_csv)
    })

    transect_id += 1

# close rasters
zsrc.close(); ssrc.close(); rsrc.close(); csrc.close()

# write index file
index_path = os.path.join(OUT_DIR, "index.csv")
pd.DataFrame(index_rows).to_csv(index_path, index=False)
print(f"Saved {len(index_rows)} transects in: {OUT_DIR}")

# -------------------------
# Plot transects over bathy
# -------------------------
if plotFlag:
    import matplotlib.pyplot as plt

    idx = pd.read_csv(index_path)

    with rasterio.open(DEPTH_TIF) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    plt.figure(figsize=(10, 10))
    plt.imshow(np.ma.masked_invalid(data), extent=extent, origin="upper")
    plt.colorbar(label="bathy (m)")

    for r in idx.itertuples(index=False):
        x0, y0 = float(r.x0), float(r.y0)
        nx, ny = float(r.nx), float(r.ny)
        L = (float(r.npoints) - 1.0) * DS_M
        x1, y1 = x0 + nx * L, y0 + ny * L

        # draw transect line
        plt.plot([x0, x1], [y0, y1], linewidth=1)

        # label it (slightly offshore so labels don't sit on the coastline)
        lab_off = 150.0  # meters along transect; adjust (0–300) if you want
        xl = x0 + nx * lab_off
        yl = y0 + ny * lab_off

        plt.text(
            xl, yl, f"{int(r.id)}",
            fontsize=10, color="k",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7)
        )


    plt.title("Transects over bathymetry")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "transects_over_bathy.png")
    plt.savefig(out_png, dpi=300)
    plt.show()
