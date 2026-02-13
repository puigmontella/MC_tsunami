#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd

# Optional plotting deps (only used if PLOT_FLAG=True)
import rasterio
import matplotlib.pyplot as plt

# -------------------------
# INPUTS (edit)
# -------------------------
IN_DIR = "transects"
OUT = os.path.join(IN_DIR, "candidate_segments.csv")

# Bathymetry GeoTIFF (for map overlay)
DEPTH_TIF = "../bati_32m_and100m_merged_EPSG32628.tif"

# -------------------------
# Candidate filter (edit freely)
# -------------------------
DEPTH_MIN = -2000   # deeper than this (more negative) -> ignore
DEPTH_MAX = -50     # shallower than this -> ignore
SLOPE_MIN = 6.0
SLOPE_MAX = 30.0 #possible rocky parts

USE_UNKNOWN_SUBSTRATE = True
MIN_LEN_M = 800.0  # minimum contiguous segment length to keep

# Roughness handling:
# Option A) Set ROUGH_MAX directly (e.g. 12.0)
# Option B) Set ROUGH_MAX = None and use AUTO_ROUGH_PCTL percentile (recommended)
ROUGH_MAX = None
AUTO_ROUGH_PCTL = 70.0  # use roughness <= percentile over all offshore points

# -------------------------
# QC / plotting controls
# -------------------------
PLOT_FLAG = True
SAVE_MAP_PNG = True
MAP_PNG = os.path.join(IN_DIR, "candidate_segments_map.png")

SAVE_PROFILE_QC = True
N_PROFILE_PLOTS = 10
PROFILE_OUT_DIR = os.path.join(IN_DIR, "qc_profiles")
os.makedirs(PROFILE_OUT_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def contiguous_true_runs(mask: np.ndarray):
    """Yield (i, j) indices for contiguous True runs in a boolean array, inclusive-exclusive."""
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        yield i, j
        i = j

def compute_auto_rough_max(tr_files, depth_min, depth_max):
    """Compute roughness percentile over points in the depth band to define ROUGH_MAX."""
    rough_all = []
    for f in tr_files:
        df = pd.read_csv(f)
        depth = df["depth"].to_numpy(float)
        rough = df["rough"].to_numpy(float)
        m = np.isfinite(depth) & np.isfinite(rough) & (depth >= depth_min) & (depth <= depth_max)
        if m.any():
            rough_all.append(rough[m])
    if not rough_all:
        return None
    rough_all = np.concatenate(rough_all)
    return float(np.nanpercentile(rough_all, AUTO_ROUGH_PCTL))

# -------------------------
# Main
# -------------------------
tr_files = sorted(glob.glob(os.path.join(IN_DIR, "tr_*.csv")))
if len(tr_files) == 0:
    raise FileNotFoundError(f"No transect CSVs found in {IN_DIR}/ (expected tr_*.csv)")

# Auto roughness threshold (if requested)
if ROUGH_MAX is None:
    rm = compute_auto_rough_max(tr_files, DEPTH_MIN, DEPTH_MAX)
    if rm is None:
        raise RuntimeError("Could not compute AUTO ROUGH_MAX (no valid roughness/depth samples).")
    ROUGH_MAX = rm
    print(f"[INFO] Auto ROUGH_MAX set to {ROUGH_MAX:.6g} (P{AUTO_ROUGH_PCTL})")

rows = []

for f in tr_files:
    df = pd.read_csv(f)

    depth = df["depth"].to_numpy(float)
    slope = df["slope_deg"].to_numpy(float)
    rough = df["rough"].to_numpy(float)
    cls   = df["cls"].to_numpy(int)
    s_m   = df["s_m"].to_numpy(float)

    ok_cls = (cls == 0) | ((cls == 2) if USE_UNKNOWN_SUBSTRATE else False)

    ok = (
        np.isfinite(depth) & np.isfinite(slope) & np.isfinite(rough) &
        (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX) &
        (slope >= SLOPE_MIN) & (slope <= SLOPE_MAX) &
        (rough <= ROUGH_MAX) &
        ok_cls
    )

    for i, j in contiguous_true_runs(ok):
        seg_len = s_m[j-1] - s_m[i]
        if seg_len < MIN_LEN_M:
            continue

        seg = df.iloc[i:j]
        rows.append({
            "transect_id": int(seg["id"].iloc[0]),
            "csv": os.path.basename(f),
            "s_start_m": float(seg["s_m"].iloc[0]),
            "s_end_m": float(seg["s_m"].iloc[-1]),
            "len_m": float(seg_len),
            "depth_mean": float(seg["depth"].mean()),
            "slope_mean": float(seg["slope_deg"].mean()),
            "rough_mean": float(seg["rough"].mean()),
            "frac_sed": float((seg["cls"] == 0).mean()),
            "npts": int(len(seg))
        })

out = pd.DataFrame(rows).sort_values(["transect_id", "s_start_m"]).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"Saved {len(out)} candidate segments -> {OUT}")

# -------------------------
# QC stats (prints)
# -------------------------
if len(out) == 0:
    print("[WARN] No candidate segments found. Consider relaxing thresholds.")
else:
    print("\n=== Candidate segments: summary stats ===")
    print(out[["len_m","depth_mean","slope_mean","rough_mean","frac_sed"]].describe())

    print("\n=== Longest 10 segments ===")
    cols = ["transect_id","len_m","depth_mean","slope_mean","rough_mean","frac_sed","csv","s_start_m","s_end_m"]
    print(out.sort_values("len_m", ascending=False).head(10)[cols].to_string(index=False))

# -------------------------
# Plot: candidate segments over bathymetry
# -------------------------
if PLOT_FLAG and SAVE_MAP_PNG and len(out) > 0:
    # cache transects needed
    transects = {}
    for csvname in out["csv"].unique():
        path = os.path.join(IN_DIR, csvname)
        transects[csvname] = pd.read_csv(path)

    with rasterio.open(DEPTH_TIF) as src:
        Z = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None:
            Z[Z == nod] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    plt.figure(figsize=(10,10))
    plt.imshow(np.ma.masked_invalid(Z), extent=extent, origin="upper")
    plt.colorbar(label="bathy (m)")

    for r in out.itertuples(index=False):
        tr = transects[r.csv]
        m = (tr["s_m"] >= r.s_start_m) & (tr["s_m"] <= r.s_end_m)
        if m.any():
            plt.plot(tr.loc[m,"x"], tr.loc[m,"y"], linewidth=2)

    plt.title("Candidate failure segments (subset of transects)")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(MAP_PNG, dpi=300)
    print(f"[INFO] Saved map -> {MAP_PNG}")
    # plt.show()  # uncomment if you want interactive

# -------------------------
# Plot: profile QC for a few segments (depth + slope)
# -------------------------
if PLOT_FLAG and SAVE_PROFILE_QC and len(out) > 0:
    rng = np.random.default_rng(42)
    pick = out.copy()

    # Prefer longer segments for QC
    pick = pick.sort_values("len_m", ascending=False).head(max(N_PROFILE_PLOTS*3, N_PROFILE_PLOTS))

    # Randomly choose N from those (or fewer if not enough)
    idxs = rng.choice(len(pick), size=min(N_PROFILE_PLOTS, len(pick)), replace=False)

    for k, ii in enumerate(idxs, start=1):
        r = pick.iloc[int(ii)]
        tr_path = os.path.join(IN_DIR, r["csv"])
        tr = pd.read_csv(tr_path)

        m = (tr["s_m"] >= r["s_start_m"]) & (tr["s_m"] <= r["s_end_m"])

        # Depth plot
        plt.figure()
        plt.plot(tr["s_m"], tr["depth"], linewidth=1)
        plt.plot(tr.loc[m,"s_m"], tr.loc[m,"depth"], linewidth=3)
     #   plt.gca().invert_yaxis()
        plt.xlabel("s (m)")
        plt.ylabel("depth (m)")
        plt.title(f"Transect {int(r.transect_id)} depth (segment highlighted)")
        outpng = os.path.join(PROFILE_OUT_DIR, f"qc_depth_tr{int(r.transect_id):04d}_seg{k:02d}.png")
        plt.tight_layout()
        plt.savefig(outpng, dpi=200)
        #plt.close()

        # # Slope plot
        # plt.figure()
        # plt.plot(tr["s_m"], tr["slope_deg"], linewidth=1)
        # plt.plot(tr.loc[m,"s_m"], tr.loc[m,"slope_deg"], linewidth=3)
        # plt.xlabel("s (m)")
        # plt.ylabel("slope (deg)")
        # plt.title(f"Transect {int(r.transect_id)} slope (segment highlighted)")
        # outpng = os.path.join(PROFILE_OUT_DIR, f"qc_slope_tr{int(r.transect_id):04d}_seg{k:02d}.png")
        # plt.tight_layout()
        # plt.savefig(outpng, dpi=200)
        # plt.close()


