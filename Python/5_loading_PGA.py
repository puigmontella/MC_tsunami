#!/usr/bin/env python3
import os
import sys
import glob
import re
import argparse

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Allow importing from project root (MC_tsunami/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utilities.geoio import load_raster_band, sample_loaded_raster
from utilities.stats import quantile_safe
from utilities.config import load_config, abspath


# -----------------------------
# Helpers
# -----------------------------
def parse_tid(path):
    base = os.path.basename(path)
    m = re.search(r"tr_(\d{1,6})_", base)
    if not m:
        m = re.search(r"tr_(\d{1,6})", base)
    if not m:
        raise ValueError(f"Cannot parse transect id from: {base}")
    return int(m.group(1))


def rp_sort_key(rp):
    try:
        return int(str(rp))
    except Exception:
        return 10**18


def raster_basic_stats_from_array(Z):
    Z = np.asarray(Z, dtype="float32")
    Z = Z[np.isfinite(Z)]
    if Z.size == 0:
        return dict(min=np.nan, max=np.nan, p01=np.nan, p50=np.nan, p99=np.nan)
    return dict(
        min=float(Z.min()),
        max=float(Z.max()),
        p01=float(np.quantile(Z, 0.01)),
        p50=float(np.quantile(Z, 0.50)),
        p99=float(np.quantile(Z, 0.99)),
    )


def raster_basic_stats(path):
    with rasterio.open(path) as src:
        Z = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None and not (isinstance(nod, float) and np.isnan(nod)):
            Z[Z == nod] = np.nan
    return raster_basic_stats_from_array(Z)


def sample_raster_at_points_chunked(src, xs, ys, chunk=200000):
    out = np.empty(len(xs), dtype=np.float32)
    out[:] = np.nan
    coords = np.column_stack([xs, ys])
    for i0 in range(0, len(coords), chunk):
        sl = slice(i0, min(len(coords), i0 + chunk))
        vals = np.array([v[0] for v in src.sample([tuple(xy) for xy in coords[sl]])], dtype=np.float32)
        out[sl] = vals
    return out.astype("float64")


def sample_pga(src_obj, xs, ys, chunk=200000):
    """
    src_obj can be:
      - dict from utilities.geoio.load_raster_band (RAM-loaded)
      - rasterio dataset
    """
    if isinstance(src_obj, dict):
        return sample_loaded_raster(src_obj, xs, ys)
    return sample_raster_at_points_chunked(src_obj, xs, ys, chunk=chunk)


def make_point_map(background_tif, x, y, c, title, out_png, cbar_label):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

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


def _mc_centers(idx_row, mc_df):
    s0 = mc_df["s0_m"].to_numpy(float)
    s1 = mc_df["s1_m"].to_numpy(float)
    s_center = 0.5 * (s0 + s1)

    x0, y0 = float(idx_row.x0), float(idx_row.y0)
    nx, ny = float(idx_row.nx), float(idx_row.ny)

    xc = x0 + nx * s_center
    yc = y0 + ny * s_center
    return xc, yc


def _compute_metrics_for_type(mc_df, idx_row, pga_srcs, rp_list, k_factor, chunk):
    """
    For one failure-type (trans or rot):
      - samples PGA at each trial center
      - computes Pf, and eta0 tail percentiles among failing trials
    Returns dict:
      { n, per_rp[rp] = {pga_mean, Pf, ef, mcount, fcount} }
    """
    need = {"s0_m", "s1_m", "kcrit", "eta0_m"}
    if (not need.issubset(mc_df.columns)) or (len(mc_df) == 0):
        return None

    xc, yc = _mc_centers(idx_row, mc_df)
    kcrit = mc_df["kcrit"].to_numpy(float)
    eta0 = mc_df["eta0_m"].to_numpy(float)

    out = {"n": int(len(mc_df)), "per_rp": {}}

    for rp in rp_list:
        src = pga_srcs.get(rp, None)
        if src is None:
            out["per_rp"][rp] = None
            continue

        pga = sample_pga(src, xc, yc, chunk=chunk) * float(k_factor)
        m = np.isfinite(pga) & np.isfinite(kcrit)

        if int(np.sum(m)) == 0:
            out["per_rp"][rp] = dict(
                pga_mean=np.nan,
                Pf=np.nan,
                ef=np.array([], float),
                mcount=0,
                fcount=0,
            )
            continue

        fail = (pga[m] >= kcrit[m])
        ef = eta0[m][fail]

        out["per_rp"][rp] = dict(
            pga_mean=float(np.nanmean(pga[m])),
            Pf=float(np.mean(fail)),
            ef=np.asarray(ef, float),
            mcount=int(np.sum(m)),
            fcount=int(np.sum(fail)),
        )

    return out


def process_one_transect(tid, idx_df, pga_srcs, rp_list, trans_path, rot_path, pcts, k_factor, chunk):
    idr = idx_df[idx_df["transect_id"] == tid]
    if len(idr) != 1:
        return None
    idr = idr.iloc[0]

    row = dict(transect_id=int(tid), x0=float(idr.x0), y0=float(idr.y0))

    info_trans = None
    info_rot = None

    if trans_path and os.path.exists(trans_path):
        mc_t = pd.read_csv(trans_path)
        info_trans = _compute_metrics_for_type(mc_t, idr, pga_srcs, rp_list, k_factor, chunk)

    if rot_path and os.path.exists(rot_path):
        mc_r = pd.read_csv(rot_path)
        info_rot = _compute_metrics_for_type(mc_r, idr, pga_srcs, rp_list, k_factor, chunk)

    row["n_trans"] = int(info_trans["n"]) if info_trans else 0
    row["n_rot"] = int(info_rot["n"]) if info_rot else 0
    row["n_any"] = row["n_trans"] + row["n_rot"]

    if row["n_any"] == 0:
        return None

    for rp in rp_list:
        # -------- trans
        if info_trans and info_trans["per_rp"].get(rp) is not None:
            d = info_trans["per_rp"][rp]
            row[f"pga{rp}_g_center_mean_trans"] = d["pga_mean"]
            row[f"Pf_{rp}_trans"] = d["Pf"]
            row[f"valid_{rp}_trans"] = d["mcount"]
            row[f"fail_{rp}_trans"] = d["fcount"]
            for p in pcts:
                row[f"eta0_fail{rp}_p{int(100*p):02d}_m_trans"] = quantile_safe(d["ef"], p)
        else:
            row[f"pga{rp}_g_center_mean_trans"] = np.nan
            row[f"Pf_{rp}_trans"] = np.nan
            row[f"valid_{rp}_trans"] = 0
            row[f"fail_{rp}_trans"] = 0
            for p in pcts:
                row[f"eta0_fail{rp}_p{int(100*p):02d}_m_trans"] = np.nan

        # -------- rot
        if info_rot and info_rot["per_rp"].get(rp) is not None:
            d = info_rot["per_rp"][rp]
            row[f"pga{rp}_g_center_mean_rot"] = d["pga_mean"]
            row[f"Pf_{rp}_rot"] = d["Pf"]
            row[f"valid_{rp}_rot"] = d["mcount"]
            row[f"fail_{rp}_rot"] = d["fcount"]
            for p in pcts:
                row[f"eta0_fail{rp}_p{int(100*p):02d}_m_rot"] = quantile_safe(d["ef"], p)
        else:
            row[f"pga{rp}_g_center_mean_rot"] = np.nan
            row[f"Pf_{rp}_rot"] = np.nan
            row[f"valid_{rp}_rot"] = 0
            row[f"fail_{rp}_rot"] = 0
            for p in pcts:
                row[f"eta0_fail{rp}_p{int(100*p):02d}_m_rot"] = np.nan

        # -------- any (concat; Pf weighted by valid counts; eta0 tails from concatenated failing set)
        ef_all = []
        valid_any = 0
        fail_any = 0
        pga_num = 0.0
        pga_den = 0.0

        if info_trans and info_trans["per_rp"].get(rp) is not None:
            d = info_trans["per_rp"][rp]
            ef_all.append(d["ef"])
            valid_any += int(d["mcount"])
            fail_any += int(d["fcount"])
            if np.isfinite(d["pga_mean"]) and d["mcount"] > 0:
                pga_num += float(d["pga_mean"]) * float(d["mcount"])
                pga_den += float(d["mcount"])

        if info_rot and info_rot["per_rp"].get(rp) is not None:
            d = info_rot["per_rp"][rp]
            ef_all.append(d["ef"])
            valid_any += int(d["mcount"])
            fail_any += int(d["fcount"])
            if np.isfinite(d["pga_mean"]) and d["mcount"] > 0:
                pga_num += float(d["pga_mean"]) * float(d["mcount"])
                pga_den += float(d["mcount"])

        ef_all = np.concatenate(ef_all) if len(ef_all) else np.array([], float)

        row[f"pga{rp}_g_center_mean_any"] = (pga_num / pga_den) if pga_den > 0 else np.nan
        row[f"Pf_{rp}_any"] = (float(fail_any) / float(valid_any)) if valid_any > 0 else np.nan
        row[f"valid_{rp}_any"] = int(valid_any)
        row[f"fail_{rp}_any"] = int(fail_any)
        for p in pcts:
            row[f"eta0_fail{rp}_p{int(100*p):02d}_m_any"] = quantile_safe(ef_all, p)

    return row


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Step 5: Sample PGA at each MC trial center and compute Pf + conditional eta0 tails for all RPs in config (trans/rot/any)."
    )
    ap.add_argument("--config", required=True, help="Path to case config.json")

    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="Parallel workers over transects (1=sequential, -1=all cores).")
    ap.add_argument("--backend", type=str, default="threading",
                    choices=["threading", "multiprocessing", "loky"],
                    help="joblib backend. threading is good when rasters are RAM-loaded (no copy).")

    ap.add_argument("--load-pga-to-ram", action="store_true",
                    help="Force load PGA rasters into RAM for faster sampling.")
    ap.add_argument("--no-load-pga-to-ram", action="store_true",
                    help="Do NOT load PGA rasters into RAM (sample via rasterio).")
    ap.add_argument("--chunk", type=int, default=200000,
                    help="Chunk size when sampling via rasterio (if not RAM-loaded).")

    ap.add_argument("--rps", default=None,
                    help="Optional comma list of RPs to process (must exist in config paths.pga_rasters). "
                         "Default: use all keys from config.")
    ap.add_argument("--k-factor", type=float, default=None,
                    help="Override k_factor (default from config pga_loading.k_factor).")

    ap.add_argument("--pcts", default="50,90,95,99",
                    help="Fail-eta percentiles (comma list, in percent).")

    ap.add_argument("--quick-maps", action="store_true",
                    help="Write quick sanity point maps for ONE RP (chosen by --map-rp or default = largest RP).")
    ap.add_argument("--map-rp", default=None,
                    help="Which RP to use for --quick-maps. Default: largest RP available.")
    ap.add_argument("--map-type", default="any", choices=["trans", "rot", "any"],
                    help="Which failure-type to map for quick maps.")
    ap.add_argument("--map-pct", type=int, default=99,
                    help="Which eta percentile to map (e.g., 90/95/99).")

    args = ap.parse_args()

    cfg = load_config(args.config)

    # ----- Paths from config
    transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"])
    index_csv = os.path.join(transects_dir, "index.csv")
    out_csv = abspath(cfg, cfg["outputs"]["summary_csv"])
    maps_dir = abspath(cfg, cfg["outputs"].get("maps_dir", transects_dir))
    bathy_tif = abspath(cfg, cfg["paths"]["bathy_tif"])

    pga_dict = cfg["paths"]["pga_rasters"]  # { "100": "path.tif", ... }

    # RPs to process
    if args.rps:
        rp_list = [x.strip() for x in str(args.rps).split(",") if x.strip() != ""]
    else:
        rp_list = sorted([str(k) for k in pga_dict.keys()], key=rp_sort_key)

    # k factor
    k_factor_cfg = float(cfg.get("pga_loading", {}).get("k_factor", 1.0))
    k_factor = float(args.k_factor) if args.k_factor is not None else k_factor_cfg

    # percentiles
    pcts = [float(x) / 100.0 for x in str(args.pcts).split(",") if x.strip() != ""]

    # RAM loading policy
    load_to_ram = True
    if args.no_load_pga_to_ram:
        load_to_ram = False
    if args.load_pga_to_ram:
        load_to_ram = True

    os.makedirs(transects_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    if not os.path.exists(index_csv):
        raise SystemExit(f"Missing index csv: {index_csv}")

    idx = pd.read_csv(index_csv).rename(columns={"id": "transect_id"})
    for c in ["transect_id", "x0", "y0", "nx", "ny"]:
        if c not in idx.columns:
            raise SystemExit(f"index.csv missing '{c}'. Columns: {list(idx.columns)}")

    # Discover MC files (prefer REGIONAL if present)
    pat_trans_reg = os.path.join(transects_dir, "tr_*_grilli_mcs_REGIONAL.csv")
    pat_trans_all = os.path.join(transects_dir, "tr_*_grilli_mcs.csv")
    pat_rot_reg = os.path.join(transects_dir, "tr_*_grilli_rot_mcs_REGIONAL.csv")
    pat_rot_all = os.path.join(transects_dir, "tr_*_grilli_rot_mcs.csv")

    trans_files = sorted(glob.glob(pat_trans_reg)) or sorted(glob.glob(pat_trans_all))
    rot_files = sorted(glob.glob(pat_rot_reg)) or sorted(glob.glob(pat_rot_all))

    trans_map = {parse_tid(f): f for f in trans_files}
    rot_map = {parse_tid(f): f for f in rot_files}

    tids = sorted(set(list(trans_map.keys()) + list(rot_map.keys())))
    if not tids:
        raise SystemExit(
            f"No MC files found in '{transects_dir}'. Expected patterns:\n"
            f"  {os.path.basename(pat_trans_reg)} / {os.path.basename(pat_trans_all)}\n"
            f"  {os.path.basename(pat_rot_reg)} / {os.path.basename(pat_rot_all)}"
        )

    # Load PGA rasters from config
    pga_srcs = {}
    print("\n=== PGA raster quick stats ===")
    for rp in rp_list:
        if rp not in pga_dict:
            print(f"[WARN] RP {rp} requested but not in config paths.pga_rasters")
            continue

        tif = abspath(cfg, pga_dict[rp])
        if not os.path.exists(tif):
            print(f"[WARN] missing PGA raster for RP={rp}: {tif}")
            continue

        if load_to_ram:
            r = load_raster_band(tif, band=1, dtype="float32")
            st = raster_basic_stats_from_array(r["arr"])
            pga_srcs[str(rp)] = r
        else:
            st = raster_basic_stats(tif)
            pga_srcs[str(rp)] = rasterio.open(tif)

        print("{:>6}  {}  min={:.4g}  p50={:.4g}  p99={:.4g}  max={:.4g}".format(
            rp, tif, st["min"], st["p50"], st["p99"], st["max"]
        ))

    if not pga_srcs:
        raise SystemExit("No PGA rasters were found/loaded. Check config paths.pga_rasters and file paths.")

    # Keep order but only those loaded
    rp_loaded = [rp for rp in rp_list if str(rp) in pga_srcs]

    print("\n=== Processing transects: n_jobs={}, backend={}, load_to_ram={}, k_factor={} ===".format(
        args.n_jobs, args.backend, load_to_ram, k_factor
    ))

    prefer = "threads" if args.backend == "threading" else "processes"
    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, prefer=prefer, verbose=10)(
        delayed(process_one_transect)(
            tid,
            idx,
            pga_srcs,
            rp_loaded,
            trans_map.get(tid, None),
            rot_map.get(tid, None),
            pcts,
            k_factor,
            int(args.chunk),
        )
        for tid in tids
    )

    rows = [r for r in results if r is not None]
    out = pd.DataFrame(rows).sort_values("transect_id").reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv}")

    # Optional quick maps (RP generic)
    if args.quick_maps and len(out) > 0:
        if not os.path.exists(bathy_tif):
            print(f"[WARN] quick maps skipped: missing bathy tif: {bathy_tif}")
        else:
            if args.map_rp is None:
                rp_map = max(rp_loaded, key=rp_sort_key)
            else:
                rp_map = str(args.map_rp).strip()

            if rp_map not in rp_loaded:
                print(f"[WARN] quick maps skipped: RP={rp_map} not loaded (loaded: {rp_loaded})")
            else:
                typ = args.map_type
                pct = int(args.map_pct)
                col_pf = f"Pf_{rp_map}_{typ}"
                col_eta = f"eta0_fail{rp_map}_p{pct:02d}_m_{typ}"

                if col_pf in out.columns:
                    make_point_map(
                        bathy_tif,
                        out["x0"], out["y0"], out[col_pf],
                        f"Pf at RP={rp_map} — {typ}",
                        os.path.join(maps_dir, f"map_Pf_RP{rp_map}_{typ}_quick.png"),
                        col_pf,
                    )
                else:
                    print(f"[INFO] quick map: missing column {col_pf}")

                if col_eta in out.columns:
                    make_point_map(
                        bathy_tif,
                        out["x0"], out["y0"], out[col_eta],
                        f"eta0 p{pct} among failures — RP={rp_map} — {typ}",
                        os.path.join(maps_dir, f"map_eta0_fail_RP{rp_map}_p{pct:02d}_{typ}_quick.png"),
                        col_eta,
                    )
                else:
                    print(f"[INFO] quick map: missing column {col_eta}")

    # Close rasters if not RAM-loaded
    if not load_to_ram:
        for src in pga_srcs.values():
            try:
                src.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
