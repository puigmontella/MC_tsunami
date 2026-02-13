import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter, distance_transform_edt
import matplotlib.pyplot as plt



# -----------------------------
# Helpers
# -----------------------------
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata
        transform = src.transform
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, profile, transform

def write_raster(path, arr, profile):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)

def fill_nans_nearest(a):
    # fill NaNs with nearest valid value
    a = a.copy()
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    a[mask] = a[tuple(idx[:, mask])]
    return a

def moving_std_nan(a, size):
    """
    NaN-aware moving window standard deviation using uniform filters + weights.
    """
    valid = np.isfinite(a).astype("float32")
    a0 = np.nan_to_num(a, nan=0.0).astype("float32")

    w = uniform_filter(valid, size=size, mode="nearest")
    mean = uniform_filter(a0, size=size, mode="nearest") / np.maximum(w, 1e-6)
    mean2 = uniform_filter(a0*a0, size=size, mode="nearest") / np.maximum(w, 1e-6)

    var = mean2 - mean*mean
    var[var < 0] = 0
    std = np.sqrt(var)
    std[w < 0.5] = np.nan  # too many NaNs in window
    return std

def slope_degrees(bathy, dx, dy):
    z = fill_nans_nearest(bathy)
    dzdy, dzdx = np.gradient(z, dy, dx)  # note: gradient order (y, x)
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    slope[np.isnan(bathy)] = np.nan
    return slope

def laplacian(bathy, dx, dy):
    z = fill_nans_nearest(bathy)
    dzdy, dzdx = np.gradient(z, dy, dx)
    d2zdy2, _ = np.gradient(dzdy, dy, dx)
    _, d2zdx2 = np.gradient(dzdx, dy, dx)
    lap = d2zdx2 + d2zdy2
    lap[np.isnan(bathy)] = np.nan
    return lap




# -----------------------------
# Inputs (edit these)
# -----------------------------
BATHY_TIF = "../bati_32m_and100m_merged_EPSG32628.tif"

# OPTIONAL (recommended): a raster aligned to bathy with labels where EMODnet exists
# e.g., rock=1, sand=2, or similar. Leave as None to use Mode B.
LABEL_TIF = None  # "emodnet_labels_aligned.tif"

# choose the roughness window (in meters) and we convert to pixels
ROUGH_WIN_M = 200  # ~10 cells at 32m; tweak 200–600m depending on what you want

# output file names
OUT_SLOPE = "slope_deg.tif"
OUT_ROUGH = "roughness_std.tif"
OUT_DEPTH = "depth.tif"


# -----------------------------
# Compute morphometrics
# -----------------------------
bathy, prof, transform = read_raster(BATHY_TIF)





from scipy.ndimage import gaussian_filter

z = fill_nans_nearest(bathy)

# Large-scale trend (remove regional slope/shape)
sigma_large_px = 14        # 14 px * 32 m ≈ 450 m  (try 12–22)
z_smooth = gaussian_filter(z, sigma=sigma_large_px)

# High-pass residual = texture
z_hp = z - z_smooth
z_hp[np.isnan(bathy)] = np.nan

# Texture roughness (std of residual)
win_small_px = 9           # 9 px * 32 m ≈ 290 m (try 7–11)
rough_tex = moving_std_nan(z_hp, size=win_small_px)




# pixel sizes
dx = abs(transform.a)
dy = abs(transform.e)


win_px = max(3, int(round(ROUGH_WIN_M / dx)))
if win_px % 2 == 0:
    win_px += 1  # odd window feels nicer

slope = slope_degrees(bathy, dx, dy)


write_raster(OUT_SLOPE, slope, prof)
write_raster(OUT_ROUGH, rough_tex, prof)



############### sediment or rock########### 
chosenCut=-2500
band = np.isfinite(bathy) & (bathy >= chosenCut)   # 0 to 2500 m depth

R95 = np.nanpercentile(rough_tex[band], 95)
R70 = np.nanpercentile(rough_tex[band], 70)

S95 = np.nanpercentile(slope[band], 95)
S70 = np.nanpercentile(slope[band], 70)


bedrock = (band & ((slope > S95) | ((slope > S70) & (rough_tex > R95)))).astype(np.uint8)
sediment = (band & ((slope < 25) & (rough_tex < R70))).astype(np.uint8)  # keep your sediment rule
unknown  = (band & (bedrock == 0) & (sediment == 0)).astype(np.uint8)


# 0 = sediment, 1 = bedrock, 2 = unknown
cls = np.full(slope.shape, 1, dtype=np.uint8)  # default unknown

cls[sediment == 1] = 0
cls[bedrock == 1] = 2

# optional: set nodata where bathy is nan (or slope/rough are nan)
cls[~(np.isfinite(slope) & np.isfinite(rough_tex))] = 3  # nodata code


# write as 0/1 masks (use your QGIS-friendly writer if you have one)
out_prof = prof.copy()
out_prof.update(dtype="uint8", count=1, nodata=3, compress="deflate")

with rasterio.open("substrate_proxy_class.tif", "w", **out_prof) as dst:
    dst.write(cls, 1)



n = band.sum()
print("Sediment %:", 100*sediment.sum()/n)
print("Bedrock %:", 100*bedrock.sum()/n)
print("Unknown %:", 100*unknown.sum()/n)
print("S70,S95:", S70, S95, " R70,R95:", R70, R95)




# ---- plot only where depth > 2000 m (i.e., bathy <= -2000) ----
deep = np.isfinite(bathy) & (bathy >= chosenCut)

# float layers: mask with NaN
rough_deep = np.where(deep, rough_tex, np.nan)
slope_deep = np.where(deep, slope, np.nan)





plt.figure()
plt.imshow(rough_deep, interpolation="nearest", cmap="jet")
plt.title("roughness (depth > 2000 m)")
plt.colorbar()
plt.clim(0, 25)

plt.figure()
plt.imshow(slope_deep, interpolation="nearest", cmap="viridis")
plt.title("slope (depth > 2000 m)")
plt.colorbar()
plt.clim(0, 25)

plt.figure()
plt.imshow(cls, interpolation="nearest", cmap="jet")
plt.title("substrate_deep (depth > 2000 m)")
plt.colorbar()
plt.clim(0, 3)




plt.show()