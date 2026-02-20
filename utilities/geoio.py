import numpy as np
import rasterio
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt

def open_raster(path):
    return rasterio.open(path)

def sample_band_at_xy(src, xs, ys, band=1, masked=True):
    """
    Fast sampling: x,y -> row,col -> array indexing.
    Returns float array with NaN for nodata/outside.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    arr = src.read(band)
    nodata = src.nodata

    rows, cols = rowcol(src.transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    out = np.full(xs.shape, np.nan, dtype="float64")

    inside = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    if not np.any(inside):
        return out

    vals = arr[rows[inside], cols[inside]].astype("float64")

    if masked and nodata is not None:
        vals = np.where(vals == nodata, np.nan, vals)

    out[inside] = vals
    return out




def load_raster_band(path, band=1, dtype="float32"):
    """
    Load a single-band raster fully into memory and return a dict
    with the array + transform + nodata + crs.
    """
    with rasterio.open(path) as src:
        arr = src.read(band).astype(dtype)
        transform = src.transform
        nodata = src.nodata
        crs = src.crs

    # convert nodata to NaN (if nodata is not already NaN)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        arr[arr == nodata] = np.nan

    return {
        "path": path,
        "arr": arr,
        "transform": transform,
        "nodata": nodata,
        "crs": crs,
        "shape": arr.shape,
    }

def sample_loaded_raster(r, xs, ys):
    """
    Sample a *loaded* raster dict (from load_raster_band) at x,y points.
    Returns float64 array with NaN for outside/invalid.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    arr = r["arr"]
    rows, cols = rowcol(r["transform"], xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    out = np.full(xs.shape, np.nan, dtype="float64")

    inside = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    if not np.any(inside):
        return out

    out[inside] = arr[rows[inside], cols[inside]].astype("float64")
    return out



 

def read_raster1(path):
    """
    Read band 1 as float32, convert nodata to NaN.
    Returns (arr, profile, transform).
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        arr[arr == nodata] = np.nan

    return arr, profile, transform

def write_raster1(path, arr, profile, nodata=np.nan, dtype="float32"):
    """
    Write a single-band raster. Keeps georeferencing from profile.
    """
    prof = profile.copy()
    prof.update(dtype=dtype, count=1, compress="deflate", nodata=nodata)

    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)

def raster_pixel_size(transform):
    dx = float(abs(transform.a))
    dy = float(abs(transform.e))
    return dx, dy


def fill_nans_nearest(a):
    """
    Fill NaNs with nearest valid value (distance transform).
    """
    a = np.asarray(a).copy()
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    a[mask] = a[tuple(idx[:, mask])]
    return a

def moving_std_nan(a, size):
    """
    NaN-aware moving window standard deviation (uniform filters + weights).
    """
    a = np.asarray(a, dtype="float32")
    valid = np.isfinite(a).astype("float32")
    a0 = np.nan_to_num(a, nan=0.0).astype("float32")

    w = uniform_filter(valid, size=int(size), mode="nearest")
    mean = uniform_filter(a0, size=int(size), mode="nearest") / np.maximum(w, 1e-6)
    mean2 = uniform_filter(a0*a0, size=int(size), mode="nearest") / np.maximum(w, 1e-6)

    var = mean2 - mean*mean
    var[var < 0] = 0
    std = np.sqrt(var)
    std[w < 0.5] = np.nan  # too many NaNs in window
    return std.astype("float32")

def slope_degrees(bathy, dx, dy):
    """
    Slope magnitude in degrees from bathymetry.
    """
    z = fill_nans_nearest(bathy)
    dzdy, dzdx = np.gradient(z, dy, dx)  # (y, x)
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype("float32")
    slope[np.isnan(bathy)] = np.nan
    return slope

def highpass_roughness_std(bathy, dx, sigma_large_px, win_small_px):
    """
    Compute roughness as std of high-pass residual:
      z_hp = z - gaussian(z, sigma_large)
      rough = std(z_hp, window)
    """
    z = fill_nans_nearest(bathy)
    z_smooth = gaussian_filter(z, sigma=float(sigma_large_px))
    z_hp = z - z_smooth
    z_hp[np.isnan(bathy)] = np.nan
    rough = moving_std_nan(z_hp, size=int(win_small_px))
    return rough



def sample_raster_points(src, xs=None, ys=None, coords=None):
    """
    Sample raster band1 at points.

    Use either:
      - xs, ys arrays
      - coords = list of (x,y)

    Returns float array with nodata -> NaN.
    """
    if coords is None:
        if xs is None or ys is None:
            raise ValueError("Provide either coords or (xs, ys).")
        coords = list(zip(xs, ys))

    vals = np.array([v[0] for v in src.sample(coords)], dtype=float)

    nd = src.nodata
    if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
        vals[vals == nd] = np.nan

    return vals
