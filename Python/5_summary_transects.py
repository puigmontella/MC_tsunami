#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd

TRANSECTS_DIR = "transects"
INDEX_CSV = os.path.join(TRANSECTS_DIR, "index.csv")
PATTERN = os.path.join(TRANSECTS_DIR, "tr_*_grilli_mcs_REGIONAL.csv")
OUT_CSV = os.path.join(TRANSECTS_DIR, "summary_regional_by_transect.csv")

# percentiles to report
PCTS = [0.05, 0.50, 0.90, 0.95, 0.99]

def q(df, col, p):
    return float(df[col].quantile(p)) if len(df) else np.nan

def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        raise SystemExit(f"No files found matching: {PATTERN}\nRun MC + regional filter first.")

    rows = []

    for f in files:
        base = os.path.basename(f)

        # base is like: tr_0003_grilli_mcs_REGIONAL.csv
        # grab the 0003 safely
        try:
            tid = int(base.split("_")[1])
        except Exception:
            # fallback: extract digits after 'tr_'
            tid = int(base.split("tr_")[1][:4])

        df = pd.read_csv(f)
        n = len(df)

        if n == 0:
            rows.append({"transect_id": tid, "n": 0})
            continue

        # helpful for diagnosing “clipping”
        s1_max = float(df["s1_m"].max())
        frac_end100 = float((df["s1_m"] >= (s1_max - 100.0)).mean())

        row = {
            "transect_id": tid,
            "n": n,

            # Stability / trigger metric
            "kcrit_mean": float(df["kcrit"].mean()),
            "kcrit_std": float(df["kcrit"].std(ddof=1)),

            # Wave proxy
            "eta0_mean_m": float(df["eta0_m"].mean()),
            "eta0_std_m": float(df["eta0_m"].std(ddof=1)),

            # Geometry / environment (good context)
            "L_mean_m": float(df["L_m"].mean()),
            "T_mean_m": float(df["T_m"].mean()),
            "beta_mean_deg": float(df["beta_deg"].mean()),
            "Ru_mean": float(df["Ru"].mean()),
            "d_center_mean_m": float(df["d_center_m"].mean()),

            # “end clipping” diagnostics
            "s1_max_m": s1_max,
            "frac_s1_within_100m_of_max": frac_end100,
        }

        # Add percentiles for key fields
        for p in PCTS:
            tag = f"p{int(round(p*100)):02d}"
            row[f"kcrit_{tag}"] = q(df, "kcrit", p)
            row[f"eta0_{tag}_m"] = q(df, "eta0_m", p)
            row[f"L_{tag}_m"] = q(df, "L_m", p)
            row[f"T_{tag}_m"] = q(df, "T_m", p)
            row[f"beta_{tag}_deg"] = q(df, "beta_deg", p)
            row[f"Ru_{tag}"] = q(df, "Ru", p)
            row[f"dcenter_{tag}_m"] = q(df, "d_center_m", p)

        row["eta0_max_m"] = float(df["eta0_m"].max())
        row["kcrit_max"] = float(df["kcrit"].max())

        rows.append(row)

    out = pd.DataFrame(rows)

    # join index.csv if available (useful for mapping later)
    if os.path.exists(INDEX_CSV):
        idx = pd.read_csv(INDEX_CSV)
        # keep only needed columns if present
        keep = [c for c in ["id","x0","y0","nx","ny","coast_s_m","npoints","csv"] if c in idx.columns]
        idx = idx[keep].rename(columns={"id": "transect_id"})
        out = out.merge(idx, on="transect_id", how="left")

    # sort by “hazard-ish” metric (eta0 tail)
    if "eta0_p99_m" in out.columns:
        out = out.sort_values("eta0_p99_m", ascending=False)
    else:
        out = out.sort_values("transect_id")

    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV}")
    print()
    print("Top 10 by eta0_p99_m:")
    cols = [c for c in ["transect_id","n","eta0_p99_m","eta0_p95_m","eta0_p50_m","kcrit_p50","kcrit_p95",
                        "beta_mean_deg","Ru_mean","d_center_mean_m","x0","y0"] if c in out.columns]
    print(out.head(10)[cols].to_string(index=False))





import rasterio
import matplotlib.pyplot as plt

TRANSECTS_DIR = "transects"
SUMMARY_CSV   = os.path.join(TRANSECTS_DIR, "summary_regional_by_transect.csv")
DEPTH_TIF     = "../../MC_tsunami_out/bati_32m_and100m_merged_EPSG32628.tif"
OUT_PNG       = os.path.join(TRANSECTS_DIR, "summary_eta0p99_map.png")

def main():
    df = pd.read_csv(SUMMARY_CSV)

    with rasterio.open(DEPTH_TIF) as src:
        Z = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None:
            Z[Z == nod] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    plt.figure(figsize=(10,10))
    plt.imshow(np.ma.masked_invalid(Z), extent=extent, origin="upper")
    plt.colorbar(label="bathy (m)")

    vals = df["eta0_p99_m"].to_numpy(float)
    sc = plt.scatter(df["x0"], df["y0"], c=vals, s=60, cmap="jet")  # no explicit colors
    plt.colorbar(sc, label="eta0_p99 (m)")

    # label top 10 (or all if you want)
    top = df.sort_values("eta0_p99_m", ascending=False).head(10)
    for r in top.itertuples(index=False):
        plt.text(r.x0, r.y0, str(int(r.transect_id)),
                 fontsize=9, ha="center", va="center")

    plt.title("Regional summary (points colored by eta0_p99)")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=250)
    plt.close()
    print(f"[OK] wrote {OUT_PNG}")

if __name__ == "__main__":
    main()
