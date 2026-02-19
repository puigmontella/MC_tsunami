#!/usr/bin/env python3
import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# -----------------------------
# EDIT PATHS IF NEEDED
# -----------------------------
TRANSECTS_DIR = "transects"
INDEX_CSV = os.path.join(TRANSECTS_DIR, "index.csv")

# Bathy for background (same CRS as transects x/y)
DEPTH_TIF = "../../MC_tsunami_out/bati_32m_and100m_merged_EPSG32628.tif"

# Prefer regional files if present
PATTERN_REGIONAL = os.path.join(TRANSECTS_DIR, "tr_*_grilli_mcs_REGIONAL.csv")
PATTERN_ALL      = os.path.join(TRANSECTS_DIR, "tr_*_grilli_mcs.csv")

OUT_DIR = os.path.join(TRANSECTS_DIR, "maps7")
OUT_SUMMARY = os.path.join(OUT_DIR, "maps7_summary.csv")

# -----------------------------
# WHAT TO MAP
# -----------------------------
# Susceptibility summaries (percentiles of kcrit)
K_PCTS = [0.05, 0.50]   # p05 and p50
ETA_PCTS = [0.95, 0.99] # p95 and p99

# Hazard-style: evaluate these pseudo-static k levels
K_LEVELS = [0.00, 0.02, 0.04, 0.06, 0.08]  # feel free to change

# Exceedance thresholds for eta0 hazard maps (meters)
ETA_THRESHOLDS = [0.5, 1.0, 2.0]  # feel free to change

# Optional: label only top-N points on some maps (0 = label none)
LABEL_TOP_N = 10

# -----------------------------
# Helpers
# -----------------------------
def parse_transect_id(path: str) -> int:
    base = os.path.basename(path)
    # matches tr_0003_...
    m = re.search(r"tr_(\d{4})", base)
    if not m:
        raise ValueError(f"Could not parse transect id from filename: {base}")
    return int(m.group(1))

def safe_quantile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))

def fmt_k(k: float) -> str:
    # nice tag: 0.02 -> k002 ; 0.1 -> k010 ; 0 -> k000
    return f"k{int(round(k*1000)):03d}"

def fmt_h(h: float) -> str:
    # 0.5 -> h0p5 ; 1.0 -> h1p0
    s = f"{h:.1f}".replace(".", "p")
    return f"h{s}"

def load_bathy_for_plot(depth_tif: str):
    with rasterio.open(depth_tif) as src:
        Z = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None:
            if not (isinstance(nod, float) and np.isnan(nod)):
                Z[Z == nod] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    return Z, extent

def plot_points_on_bathy(Z, extent, x, y, values, title, outpng,
                         label_ids=None, label_x=None, label_y=None):
    plt.figure(figsize=(10, 10))
    im = plt.imshow(np.ma.masked_invalid(Z), extent=extent, origin="upper")
    plt.colorbar(im, label="bathy (m)")

    sc = plt.scatter(x, y, c=values, s=60)
    plt.colorbar(sc, label=title)

    if label_ids is not None and label_x is not None and label_y is not None:
        for tid, xx, yy in zip(label_ids, label_x, label_y):
            plt.text(xx, yy, str(int(tid)), fontsize=9, ha="center", va="center")

    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(outpng, dpi=250)
    plt.close()
    print(f"[OK] wrote {outpng}")

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INDEX_CSV):
        raise SystemExit(f"Missing {INDEX_CSV}. Run transect maker first.")

    idx = pd.read_csv(INDEX_CSV)
    need_cols = {"id", "x0", "y0"}
    if not need_cols.issubset(idx.columns):
        raise SystemExit(f"index.csv missing columns {need_cols}. Found: {list(idx.columns)}")

    idx = idx.rename(columns={"id": "transect_id"})[["transect_id", "x0", "y0"]].copy()
    idx["transect_id"] = idx["transect_id"].astype(int)

    # Find MC files
    files = sorted(glob.glob(PATTERN_REGIONAL))
    using_regional = True
    if not files:
        files = sorted(glob.glob(PATTERN_ALL))
        using_regional = False

    if not files:
        raise SystemExit("No MC output files found. Expected *_grilli_mcs(_REGIONAL).csv in transects/")

    print(f"[INFO] Using {'REGIONAL' if using_regional else 'ALL'} MC files: {len(files)} transects")

    rows = []
    for f in files:
        tid = parse_transect_id(f)
        df = pd.read_csv(f)

        if df.empty:
            continue

        if not {"kcrit", "eta0_m"}.issubset(df.columns):
            raise SystemExit(f"File {f} missing required columns. Has: {list(df.columns)}")

        kcrit = df["kcrit"].to_numpy(float)
        eta0  = df["eta0_m"].to_numpy(float)

        # Basic summaries
        row = {
            "transect_id": tid,
            "n": int(len(df)),
        }

        for p in K_PCTS:
            row[f"kcrit_p{int(p*100):02d}"] = safe_quantile(kcrit, p)
        for p in ETA_PCTS:
            row[f"eta0_p{int(p*100):02d}_m"] = safe_quantile(eta0, p)

        row["kcrit_mean"] = float(np.nanmean(kcrit))
        row["eta0_mean_m"] = float(np.nanmean(eta0))

        # Hazard-style metrics at each k level
        for k in K_LEVELS:
            mk = np.isfinite(kcrit) & (kcrit <= k)
            Pf = float(mk.mean()) if mk.size else np.nan
            row[f"Pf_{fmt_k(k)}"] = Pf

            # exceedance hazards P(eta0 >= h AND fail at k)
            for h in ETA_THRESHOLDS:
                m = mk & np.isfinite(eta0) & (eta0 >= h)
                row[f"Peta_ge_{fmt_h(h)}_{fmt_k(k)}"] = float(m.mean()) if mk.size else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No non-empty transect MC files found.")

    # Merge coordinates
    out = out.merge(idx, on="transect_id", how="left")
    out.to_csv(OUT_SUMMARY, index=False)
    print(f"[OK] wrote {OUT_SUMMARY}")

    # Load bathy background
    Zb, extent = load_bathy_for_plot(DEPTH_TIF)

    # Decide which points to label (top by eta0_p99 if present)
    label_ids = label_x = label_y = None
    if LABEL_TOP_N and LABEL_TOP_N > 0 and "eta0_p99_m" in out.columns:
        top = out.sort_values("eta0_p99_m", ascending=False).head(LABEL_TOP_N)
        label_ids = top["transect_id"].to_numpy()
        label_x   = top["x0"].to_numpy()
        label_y   = top["y0"].to_numpy()

    # -----------------------------
    # MAPS: susceptibility
    # -----------------------------
    for p in K_PCTS:
        col = f"kcrit_p{int(p*100):02d}"
        if col in out.columns:
            outpng = os.path.join(OUT_DIR, f"map_{col}.png")
            plot_points_on_bathy(
                Zb, extent,
                out["x0"].to_numpy(float),
                out["y0"].to_numpy(float),
                out[col].to_numpy(float),
                title=f"{col} (susceptibility)",
                outpng=outpng,
                label_ids=label_ids, label_x=label_x, label_y=label_y
            )

    for p in ETA_PCTS:
        col = f"eta0_p{int(p*100):02d}_m"
        if col in out.columns:
            outpng = os.path.join(OUT_DIR, f"map_{col}.png")
            plot_points_on_bathy(
                Zb, extent,
                out["x0"].to_numpy(float),
                out["y0"].to_numpy(float),
                out[col].to_numpy(float),
                title=f"{col} (unconditional tail)",
                outpng=outpng,
                label_ids=label_ids, label_x=label_x, label_y=label_y
            )

    # -----------------------------
    # MAPS: hazard-style (Pf(k) and exceedance)
    # -----------------------------
    for k in K_LEVELS:
        col = f"Pf_{fmt_k(k)}"
        if col in out.columns:
            outpng = os.path.join(OUT_DIR, f"map_{col}.png")
            plot_points_on_bathy(
                Zb, extent,
                out["x0"].to_numpy(float),
                out["y0"].to_numpy(float),
                out[col].to_numpy(float),
                title=f"{col}  (Pf at k={k:g})",
                outpng=outpng,
                label_ids=label_ids, label_x=label_x, label_y=label_y
            )

        for h in ETA_THRESHOLDS:
            col2 = f"Peta_ge_{fmt_h(h)}_{fmt_k(k)}"
            if col2 in out.columns:
                outpng = os.path.join(OUT_DIR, f"map_{col2}.png")
                plot_points_on_bathy(
                    Zb, extent,
                    out["x0"].to_numpy(float),
                    out["y0"].to_numpy(float),
                    out[col2].to_numpy(float),
                    title=f"{col2}  (P(eta0≥{h:g}m AND fail@k={k:g}))",
                    outpng=outpng,
                    label_ids=label_ids, label_x=label_x, label_y=label_y
                )

    print(f"[DONE] Maps written in: {OUT_DIR}")

if __name__ == "__main__":
    main()
