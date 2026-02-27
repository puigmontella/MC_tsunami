import numpy as np

def assert_projected(crs):
    if crs is None or not crs.is_projected:
        raise ValueError("CRS must be projected (meters), not geographic (degrees).")

def assert_same_grid(src_a, src_b, name_a="A", name_b="B"):
    if src_a.crs != src_b.crs:
        raise ValueError("CRS mismatch {} vs {}".format(name_a, name_b))
    if src_a.transform != src_b.transform:
        raise ValueError("Transform mismatch {} vs {}".format(name_a, name_b))
    if src_a.width != src_b.width or src_a.height != src_b.height:
        raise ValueError("Shape mismatch {} vs {}".format(name_a, name_b))

def check_bathy_sign(depth_arr, expect_negative=True):
    depth_arr = np.asarray(depth_arr)
    depth_arr = depth_arr[np.isfinite(depth_arr)]
    if depth_arr.size == 0:
        raise ValueError("No finite bathymetry values to check.")
    med = float(np.median(depth_arr))
    if expect_negative and med > 0:
        raise ValueError("Bathymetry seems positive (median={}). Expected negative below sea level.".format(med))
