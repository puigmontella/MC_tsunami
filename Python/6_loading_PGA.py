#!/usr/bin/env python3
import os, glob, re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# -----------------------------
# EDIT THESE PATHS
# -----------------------------
TRANSECTS_DIR = "transects"
INDEX_CSV     = os.path.join(TRANSECTS_DIR, "index.csv")

# MC outputs (regional)
MC_PATTERN    = os.path.join(TRANSECTS_DIR, "tr_*_grilli_mcs_REGIONAL.csv")

# PGA rasters (same CRS/grid as bathy)
PGA_RASTERS = {
    "100":  "../../MC_tsunami_out/pga_100.tif",
    "200":  "../../MC_tsunami_out/pga_200.tif",
    "475":  "../../MC_tsunami_out/pga_475.tif",
    "2475": "../../MC_tsunami_out/pga_2475.tif",
}

# Optional bathy background for point maps
BATHY_TIF = "../../MC_tsunami_out/bati_32m_and100m_merged_EPSG32628.tif"

OUT_CSV   = os.path.join(TRANSECTS_DIR, "summary_eq_center_all_rp.csv")
OUT_DIR_MAPS   = TRANSECTS_DIR
OUT_DIR_CURVES = os.path.join(TRANSECTS_DIR, "curves")
os.makedirs(OUT_DIR_CURVES, exist_ok=True)

# If you want to reduce PGA to an effective pseudo-static coefficient:
# common choices: 0.5*PGA, 0.65*PGA, or PGA itself.
K_FACTOR = 1.0

# Curves: which transects to plot (or set None to plot top 6 by eta0_fail475_p99 later)
PLOT_TIDS = None  # e.g. [7, 13, 11, 0]

# Curve grid (PGA/k axis)
K_MAX = 0.15
N_K   = 120

# Chunked raster sampling (memory safe)
CHUNK = 200000

PCTS = [0.50, 0.90, 0.95, 0.99]

def parse_tid(path):
    base = os.path.basename(path)
    m = re.search(r"tr_(\d+)_", base)
    if not m:
        raise ValueError(f"Cannot parse transect id from: {base}")
    return int(m.group(1))

def q(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.quantile(a, p))

def sample_raster_at_points(src, xs, ys, chunk=200000):
    out = np.empty(len(xs), dtype=np.float32)
    out[:] = np.nan
    coords = np.column_stack([xs, ys])
    for i0 in range(0, len(coords), chunk):
        sl = slice(i0, min(len(coords), i0 + chunk))
        vals = np.array([v[0] for v in src.sample([tuple(xy) for xy in coords[sl]])], dtype=np.float32)
        out[sl] = vals
    return out

def raster_basic_stats(path):
    with rasterio.open(path) as src:
        Z = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None and not (isinstance(nod, float) and np.isnan(nod)):
            Z[Z == nod] = np.nan
        Z = Z[np.isfinite(Z)]
        if Z.size == 0:
            return dict(min=np.nan, max=np.nan, p01=np.nan, p50=np.nan, p99=np.nan)
        return dict(min=float(Z.min()), max=float(Z.max()),
                    p01=float(np.quantile(Z, 0.01)), p50=float(np.quantile(Z, 0.50)), p99=float(np.quantile(Z, 0.99)))

def make_point_map(background_tif, x, y, c, title, out_png, cbar_label):
    # background is optional
    plt.figure(figsize=(10, 10))
    if background_tif and os.path.exists(background_tif):
        with rasterio.open(background_tif) as src:
            Z = src.read(1).astype("float32")
            nod = src.nodata
            if nod is not None and not (isinstance(nod, float) and np.isnan(nod)):
                Z[Z == nod] = np.nan
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        plt.imshow(np.ma.masked_invalid(Z), extent=extent, origin="upper")
        plt.colorbar(label="bathy (m)")
    sc = plt.scatter(x, y, c=c, s=50, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, label=cbar_label)
    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[OK] wrote {out_png}")

def plot_curves_for_transect(tid, idx_row, mc_df, pga_srcs):
    """
    Produces 2 plots:
      1) Pf(k) curve from kcrit distribution (no PGA raster needed)
      2) Pf(k) and eta0_p99(k) using *center-sampled* PGA raster at 475 (optional but informative)
    """
    kcrit = mc_df["kcrit"].to_numpy(float)
    eta0  = mc_df["eta0_m"].to_numpy(float)
    s0 = mc_df["s0_m"].to_numpy(float)
    s1 = mc_df["s1_m"].to_numpy(float)
    s_center = 0.5*(s0+s1)

    x0, y0, nx, ny = float(idx_row.x0), float(idx_row.y0), float(idx_row.nx), float(idx_row.ny)
    xc = x0 + nx*s_center
    yc = y0 + ny*s_center

    # Pf(k) purely from kcrit distribution: Pf = P(k >= kcrit)
    kgrid = np.linspace(0.0, K_MAX, N_K)
    kcrit_f = kcrit[np.isfinite(kcrit)]
    kcrit_sorted = np.sort(kcrit_f)
    # ECDF: P(kcrit <= k) => Pf(k)=that
    pf_k = np.searchsorted(kcrit_sorted, kgrid, side="right") / max(1, len(kcrit_sorted))

    # Plot 1: Pf(k) curve
    plt.figure()
    plt.plot(kgrid, pf_k, linewidth=2)
    plt.xlabel("k (≈ PGA in g units)")
    plt.ylabel("Pf(k) = P(fail | k)")
    plt.title(f"Transect {tid:04d} — failure curve from kcrit distribution")
    plt.grid(True, alpha=0.3)
    out1 = os.path.join(OUT_DIR_CURVES, f"tr_{tid:04d}_Pf_kcrit_curve.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    # Plot 2: conditional eta0 tail vs k (uses actual eta0 values, still driven by kcrit thresholding)
    # eta0_p99(k) among failures defined by (k >= kcrit)
    eta0_p99 = []
    for k in kgrid:
        m = np.isfinite(eta0) & np.isfinite(kcrit) & (k >= kcrit)
        eta0_p99.append(q(eta0[m], 0.99))
    eta0_p99 = np.array(eta0_p99, float)

    plt.figure()
    plt.plot(kgrid, eta0_p99, linewidth=2)
    plt.xlabel("k (≈ PGA in g units)")
    plt.ylabel("eta0_p99 among failing trials (m)")
    plt.title(f"Transect {tid:04d} — eta0 tail vs k threshold")
    plt.grid(True, alpha=0.3)
    out2 = os.path.join(OUT_DIR_CURVES, f"tr_{tid:04d}_eta0p99_vs_k.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    # Optional: show how PGA varies along slide centers (for 475 only)
    if "475" in pga_srcs:
        pga475 = sample_raster_at_points(pga_srcs["475"], xc, yc, chunk=CHUNK).astype(float) * K_FACTOR
        # scatter (kcrit vs pga475) just for sanity
        m = np.isfinite(pga475) & np.isfinite(kcrit)
        plt.figure()
        plt.scatter(kcrit[m], pga475[m], s=2, alpha=0.2)
        plt.xlabel("kcrit")
        plt.ylabel("PGA475(center) [g]")
        plt.title(f"Transect {tid:04d} — center PGA475 vs kcrit (sanity)")
        plt.grid(True, alpha=0.3)
        out3 = os.path.join(OUT_DIR_CURVES, f"tr_{tid:04d}_PGA475center_vs_kcrit.png")
        plt.tight_layout()
        plt.savefig(out3, dpi=200)
        plt.close()

    print(f"[OK] wrote curves for transect {tid}")

def main():
    # Load index
    if not os.path.exists(INDEX_CSV):
        raise SystemExit(f"Missing {INDEX_CSV}")
    idx = pd.read_csv(INDEX_CSV).rename(columns={"id": "transect_id"})
    for c in ["transect_id", "x0", "y0", "nx", "ny"]:
        if c not in idx.columns:
            raise SystemExit(f"index.csv missing '{c}'. Columns: {list(idx.columns)}")

    # Find MC files
    files = sorted(glob.glob(MC_PATTERN))
    if not files:
        raise SystemExit(f"No MC files found: {MC_PATTERN}")

    # Open rasters once
    pga_srcs = {}
    print("\n=== PGA raster quick stats (check for 'constant' maps) ===")
    for rp, path in PGA_RASTERS.items():
        if not os.path.exists(path):
            print(f"[WARN] missing {rp} raster: {path}")
            continue
        st = raster_basic_stats(path)
        print(f"{rp:>4}  {path}  min={st['min']:.4g}  p50={st['p50']:.4g}  p99={st['p99']:.4g}  max={st['max']:.4g}")
        pga_srcs[rp] = rasterio.open(path)

    rows = []
    # per transect
    for f in files:
        tid = parse_tid(f)
        idr = idx[idx["transect_id"] == tid]
        if len(idr) != 1:
            print(f"[WARN] transect {tid} not found uniquely in index.csv -> skipping")
            continue
        idr = idr.iloc[0]

        mc = pd.read_csv(f)
        need = {"s0_m","s1_m","kcrit","eta0_m"}
        if not need.issubset(mc.columns):
            print(f"[WARN] {os.path.basename(f)} missing {need - set(mc.columns)} -> skipping")
            continue

        n = len(mc)
        if n == 0:
            continue

        s_center = 0.5*(mc["s0_m"].to_numpy(float) + mc["s1_m"].to_numpy(float))
        x0, y0, nx, ny = float(idr.x0), float(idr.y0), float(idr.nx), float(idr.ny)
        xc = x0 + nx*s_center
        yc = y0 + ny*s_center

        kcrit = mc["kcrit"].to_numpy(float)
        eta0  = mc["eta0_m"].to_numpy(float)

        row = dict(transect_id=tid, n=n, x0=x0, y0=y0)

        # For each RP raster, sample PGA at center points (per trial)
        for rp, src in pga_srcs.items():
            pga = sample_raster_at_points(src, xc, yc, chunk=CHUNK).astype(float) * K_FACTOR
            m = np.isfinite(pga) & np.isfinite(kcrit)

            if m.sum() == 0:
                row[f"pga{rp}_g_center_mean"] = np.nan
                row[f"Pf_{rp}"] = np.nan
                for p in PCTS:
                    row[f"eta0_fail{rp}_p{int(100*p):02d}_m"] = np.nan
                continue

            row[f"pga{rp}_g_center_mean"] = float(np.nanmean(pga[m]))
            fail = m & (pga >= kcrit)
            row[f"Pf_{rp}"] = float(fail.sum() / m.sum())

            ef = eta0[fail]
            for p in PCTS:
                row[f"eta0_fail{rp}_p{int(100*p):02d}_m"] = q(ef, p)

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("transect_id").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] wrote {OUT_CSV}")

    # Simple maps (points)
    if "475" in pga_srcs and len(out) > 0:
        make_point_map(BATHY_TIF, out["x0"], out["y0"], out["Pf_475"],
                       "Pf at 475y (center-sampled PGA)", os.path.join(OUT_DIR_MAPS, "map_Pf475_center.png"),
                       "Pf_475")
        make_point_map(BATHY_TIF, out["x0"], out["y0"], out["eta0_fail475_p99_m"],
                       "eta0_p99 among failures (475y, center PGA)", os.path.join(OUT_DIR_MAPS, "map_eta0_fail475_p99_center.png"),
                       "eta0_fail475_p99 (m)")

    # Curves (choose transects)
    if len(out) > 0:
        if PLOT_TIDS is None:
            # pick top 6 by eta0_fail475_p99 if present, else by transect id
            col = "eta0_fail475_p99_m"
            if col in out.columns:
                tids = out.sort_values(col, ascending=False).head(6)["transect_id"].tolist()
            else:
                tids = out["transect_id"].head(6).tolist()
        else:
            tids = list(PLOT_TIDS)

        for tid in tids:
            f = os.path.join(TRANSECTS_DIR, f"tr_{tid:04d}_grilli_mcs_REGIONAL.csv")
            if not os.path.exists(f):
                continue
            mc = pd.read_csv(f)
            idr = idx[idx["transect_id"] == tid].iloc[0]
            plot_curves_for_transect(tid, idr, mc, pga_srcs)

    # close rasters
    for src in pga_srcs.values():
        src.close()

if __name__ == "__main__":
    main()
