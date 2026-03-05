#!/usr/bin/env python3
"""
7_maps_and_figures.py

Creates figures (alongshore curves + maps) from Monte Carlo pipeline outputs.

Outputs are organized into:
  - <maps_dir>/runup/
  - <maps_dir>/failure/

RUNUP inputs (from step 6):
  - <coast_dir>/coast_runup_eqRP{eqrp}_{type}_{source}.csv
      type   in {trans, rot, any}
      source in {transects, areas, both}
  - legacy supported:
      <coast_dir>/coast_runup_eqRP{eqrp}_{type}.csv  (treated as source="both")

FAILURE inputs (from MC steps):
  Transects:
    - <transects_dir>/tr_*_trans_mcs.csv
    - <transects_dir>/tr_*_rot_mcs.csv
  Areas:
    - <areas_dir>/ar_*_trans_mcs.csv
    - <areas_dir>/ar_*_rot_mcs.csv

Background raster:
  - <rasters_dir>/depth_pos.tif

Index (coast geometry + alongshore distance):
  - <transects_dir>/index.csv
"""

import os
import json
import glob
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform as rio_transform
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Colormaps (matplotlib names)
# -----------------------------
CMAP_DEPTH = "Greys_r"
CMAP_RUNUP = "turbo"
CMAP_LAMBDA = "viridis"
CMAP_PF = "magma"
CMAP_VOL = "turbo"
CMAP_AREA = "turbo"


# -----------------------------
# Formatting: degrees + minutes
# -----------------------------
def degmin(val, kind="lon"):
    if not np.isfinite(val):
        return "nan"
    v = float(val)
    av = abs(v)
    deg = int(av)
    minutes = (av - deg) * 60.0
    if kind == "lon":
        hemi = "E" if v >= 0 else "W"
        return f"{deg:d}°{minutes:02.0f}'{hemi}"
    else:
        hemi = "N" if v >= 0 else "S"
        return f"{deg:d}°{minutes:02.0f}'{hemi}"


# -----------------------------
# Core plotting: map on raster
# -----------------------------
def scatter_on_depth(depth, extent, x, y, c, cmap, c_label, title, out_png,
                     src_crs=None, depth_vmin=None, depth_vmax=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.asarray(c, float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if not np.any(m):
        return False

    cmin = float(np.nanmin(c[m]))
    cmax = float(np.nanmax(c[m]))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
        cmin, cmax = 0.0, 1.0

    fig, ax = plt.subplots()

    im = ax.imshow(
        depth,
        extent=extent,
        origin="upper",
        cmap=CMAP_DEPTH,
        vmin=depth_vmin,
        vmax=depth_vmax,
        interpolation="nearest",
        alpha=0.85
    )

    sc = ax.scatter(
        x[m], y[m],
        c=c[m],
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        s=45,
        linewidths=0.2,
        edgecolors="k",
    )

    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.2)

    if src_crs is not None:
        left, right, bottom, top = extent
        xmid = 0.5 * (left + right)
        ymid = 0.5 * (bottom + top)

        def fmt_lon(xv, pos):
            lons, _ = rio_transform(src_crs, "EPSG:4326", [float(xv)], [float(ymid)])
            return degmin(lons[0], "lon")

        def fmt_lat(yv, pos):
            _, lats = rio_transform(src_crs, "EPSG:4326", [float(xmid)], [float(yv)])
            return degmin(lats[0], "lat")

        ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
            lab.set_ha("right")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    divider = make_axes_locatable(ax)

    cax_depth = divider.append_axes("bottom", size="6%", pad=0.55)
    cb_depth = fig.colorbar(im, cax=cax_depth, orientation="horizontal")
    cb_depth.set_label("Depth [m]")

    cax_data = divider.append_axes("right", size="4%", pad=0.15)
    cb_data = fig.colorbar(sc, cax=cax_data)
    cb_data.set_label(c_label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


# -----------------------------
# Failure helpers
# -----------------------------
def nearest_coast_id(x, y, x_coast, y_coast, coast_ids, coast_s):
    dx = x_coast - float(x)
    dy = y_coast - float(y)
    j = int(np.argmin(dx*dx + dy*dy))
    return int(coast_ids[j]), float(coast_s[j])


def failure_stats_from_df(df):
    """
    Needs: L_eff_m, w_m, T_m, failed_within_cap
    Returns:
      n, nfail, Pf, Vfail, Afail
    """
    need = {"L_eff_m", "w_m", "T_m", "failed_within_cap"}
    if not need.issubset(df.columns):
        return dict(n=0, nfail=0, Pf=np.nan, Vfail=np.array([], float), Afail=np.array([], float))

    L = pd.to_numeric(df["L_eff_m"], errors="coerce").to_numpy(float)
    W = pd.to_numeric(df["w_m"], errors="coerce").to_numpy(float)
    T = pd.to_numeric(df["T_m"], errors="coerce").to_numpy(float)
    F = pd.to_numeric(df["failed_within_cap"], errors="coerce").to_numpy(float)

    ok = np.isfinite(L) & np.isfinite(W) & np.isfinite(T) & np.isfinite(F)
    if not np.any(ok):
        return dict(n=0, nfail=0, Pf=np.nan, Vfail=np.array([], float), Afail=np.array([], float))

    L = L[ok]; W = W[ok]; T = T[ok]; F = F[ok]
    A = L * W
    V = A * T

    n = int(len(F))
    mfail = (F >= 0.5) & np.isfinite(V) & np.isfinite(A)
    nfail = int(np.sum(mfail))
    Pf = float(nfail / n) if n > 0 else np.nan
    return dict(n=n, nfail=nfail, Pf=Pf, Vfail=V[mfail], Afail=A[mfail])


def pct_or_nan(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.nanpercentile(x, p))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--types", default=None,
                    help="Runup types: comma list among trans,rot,any. Default: all available.")
    ap.add_argument("--sources", default=None,
                    help="Runup sources: comma list among transects,areas,both. Default: all available.")
    ap.add_argument("--eq-rps", default=None,
                    help="Comma list: 100,475,2475,... Default: all available.")
    ap.add_argument("--map-Ts", default=None,
                    help="Comma list: 100,200,2475. Default: 100,500,2475.")
    ap.add_argument("--fail-mechs", default=None,
                    help="Failure mechs: comma list among trans,rot. Default: trans,rot.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    transects_dir = cfg["outputs"]["transects_dir"]
    coast_dir = cfg["outputs"]["coast_dir"]
    maps_dir = cfg["outputs"]["maps_dir"]
    rasters_dir = cfg["outputs"]["rasters_dir"]
    areas_dir = cfg["outputs"].get("areas_dir", "outputs/areas")

    runup_dir = os.path.join(maps_dir, "runup")
    failure_dir = os.path.join(maps_dir, "failure")
    os.makedirs(runup_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    nameCase = os.path.basename(os.getcwd())

    # ==========================================================
    # A) Load background raster (depth)
    # ==========================================================
    depth_tif = os.path.join(rasters_dir, "depth_pos.tif")
    if not os.path.exists(depth_tif):
        raise SystemExit(f"[FAIL] Missing depth raster: {depth_tif}")

    with rasterio.open(depth_tif) as ds:
        depth = ds.read(1).astype(float)
        depth[~np.isfinite(depth)] = np.nan
        b = ds.bounds
        extent = (b.left, b.right, b.bottom, b.top)
        src_crs = ds.crs

    dm = np.isfinite(depth)
    depth_vmin = float(np.nanmin(depth[dm])) if np.any(dm) else None
    depth_vmax = float(np.nanmax(depth[dm])) if np.any(dm) else None

    # ==========================================================
    # B) Load index.csv
    # ==========================================================
    idx_path = os.path.join(transects_dir, "index.csv")
    if not os.path.exists(idx_path):
        raise SystemExit(f"[FAIL] Missing index.csv: {idx_path}")

    idx = pd.read_csv(idx_path)
    if "id" in idx.columns:
        idx = idx.rename(columns={"id": "transect_id"})
    for c in ("transect_id", "x0", "y0", "coast_s_m"):
        if c not in idx.columns:
            raise SystemExit(f"[FAIL] index.csv missing required column: {c}")

    idx["transect_id"] = pd.to_numeric(idx["transect_id"], errors="coerce").astype("Int64")
    if not idx["coast_s_m"].notna().any():
        raise SystemExit("[FAIL] index.csv must contain coast_s_m for alongshore plots.")

    coast_ids = pd.to_numeric(idx["transect_id"], errors="coerce").to_numpy(int)
    x_coast = pd.to_numeric(idx["x0"], errors="coerce").to_numpy(float)
    y_coast = pd.to_numeric(idx["y0"], errors="coerce").to_numpy(float)
    coast_s = pd.to_numeric(idx["coast_s_m"], errors="coerce").to_numpy(float)

    # ==========================================================
    # C) Parse filters
    # ==========================================================
    want_types = None
    if args.types:
        want_types = set(t.strip().lower() for t in args.types.split(",") if t.strip())

    want_sources = None
    if args.sources:
        want_sources = set(s.strip().lower() for s in args.sources.split(",") if s.strip())

    want_eqrps = None
    if args.eq_rps:
        want_eqrps = set(int(x.strip()) for x in args.eq_rps.split(",") if x.strip())

    Ts = [100, 500, 2475]
    if args.map_Ts:
        Ts = [int(x.strip()) for x in args.map_Ts.split(",") if x.strip()]

    fail_mechs = ["trans", "rot"]
    if args.fail_mechs:
        fail_mechs = [m.strip().lower() for m in args.fail_mechs.split(",") if m.strip()]

    # ==========================================================
    # 1) RUNUP FIGURES
    # ==========================================================
    coast_files = sorted(glob.glob(os.path.join(coast_dir, "coast_runup_eqRP*_*.csv")))
    if not coast_files:
        print(f"[WARN] No coast_runup_eqRP*_*.csv found in: {coast_dir}")
    else:
        # Parse: eqrp, type, source(optional)
        # new: coast_runup_eqRP100_trans_areas.csv
        # legacy: coast_runup_eqRP100_trans.csv  -> source="both"
        rx_new = re.compile(r"coast_runup_eqRP(\d+)_([a-zA-Z]+)_([a-zA-Z]+)\.csv$")
        rx_old = re.compile(r"coast_runup_eqRP(\d+)_([a-zA-Z]+)\.csv$")

        entries = []
        for fcsv in coast_files:
            base = os.path.basename(fcsv)
            m = rx_new.search(base)
            if m:
                eqrp = int(m.group(1))
                typ = m.group(2).lower()
                src = m.group(3).lower()
                entries.append((eqrp, typ, src, fcsv, False))
                continue
            m = rx_old.search(base)
            if m:
                eqrp = int(m.group(1))
                typ = m.group(2).lower()
                src = "both"
                entries.append((eqrp, typ, src, fcsv, True))

        # Prefer explicit (non-legacy) if duplicates exist
        best = {}
        for eqrp, typ, src, path, is_legacy in entries:
            key = (eqrp, typ, src)
            if key not in best:
                best[key] = (path, is_legacy)
            else:
                # keep non-legacy if present
                if best[key][1] and (not is_legacy):
                    best[key] = (path, is_legacy)

        keys = sorted(best.keys())

        for (eqrp, typ, src) in keys:
            print ("want_eqrps", want_eqrps)
            print ("want_types", want_types)
            print ("want_sources", want_sources)

            print ("eqrp", eqrp)
            print ("typ", typ)
            print ("src", src)
            if want_eqrps and eqrp not in want_eqrps:
                continue
            if want_types and typ not in want_types:
                continue
            if want_sources and src not in want_sources:
                continue

            fcsv = best[(eqrp, typ, src)][0]

            mech_lbl = {"trans": "translational", "rot": "rotational", "any": "combined"}.get(typ, typ)
            src_lbl = {"transects": "transects", "areas": "areas", "both": "transects+areas"}.get(src, src)

            df = pd.read_csv(fcsv)
            required = {"coast_id", "x", "y", "lambda_RunUp_ge_thr_1yr"}
            for T in Ts:
                required.add(f"RunUp_{T}yr")
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[WARN] skipping {os.path.basename(fcsv)} (missing columns: {missing})")
                continue

            df["coast_id"] = pd.to_numeric(df["coast_id"], errors="coerce").astype("Int64")
            df["x"] = pd.to_numeric(df["x"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")

            tmp = idx[["transect_id", "coast_s_m"]].rename(columns={"transect_id": "coast_id"})
            df = df.merge(tmp, on="coast_id", how="left")
            if not df["coast_s_m"].notna().any():
                print(f"[WARN] could not map coast_s_m onto {os.path.basename(fcsv)} (coast_id mismatch?)")
                continue

            df = df.sort_values("coast_s_m").reset_index(drop=True)

            # ---- Alongshore curves (this file) ----
            plt.figure()
            xa = pd.to_numeric(df["coast_s_m"], errors="coerce").to_numpy(float)
            for T in Ts:
                yv = pd.to_numeric(df[f"RunUp_{T}yr"], errors="coerce").to_numpy(float)
                plt.plot(xa, yv, marker="o", linewidth=1.5, markersize=3, label=f"RunUp_{T}yr [m]")
            plt.xlabel("Alongshore distance coast_s_m [m]")
            plt.ylabel("Runup [m]")
            plt.title(f"{nameCase} — Design runup alongshore\nEQ RP = {eqrp} yr | Type = {mech_lbl} | Source = {src_lbl}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(runup_dir, f"runup_alongshore_eqRP{eqrp}_{typ}_{src}.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print("[OK]", out_png)

            # ---- Maps ----
            X = df["x"].to_numpy(float)
            Y = df["y"].to_numpy(float)

            for T in Ts:
                ccol = f"RunUp_{T}yr"
                C = pd.to_numeric(df[ccol], errors="coerce").to_numpy(float)
                out_png = os.path.join(runup_dir, f"runup_map_eqRP{eqrp}_{typ}_{src}_T{T}.png")
                ok = scatter_on_depth(
                    depth, extent, X, Y, C,
                    cmap=CMAP_RUNUP,
                    c_label=f"{ccol} [m]",
                    title=f"{nameCase} — Runup map\nEQ RP = {eqrp} yr | Type = {mech_lbl} | Source = {src_lbl} | Tsunami T = {T} yr",
                    out_png=out_png,
                    src_crs=src_crs,
                    depth_vmin=depth_vmin, depth_vmax=depth_vmax
                )
                if ok:
                    print("[OK]", out_png)

            C = pd.to_numeric(df["lambda_RunUp_ge_thr_1yr"], errors="coerce").to_numpy(float)
            out_png = os.path.join(runup_dir, f"lambda_map_eqRP{eqrp}_{typ}_{src}.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, C,
                cmap=CMAP_LAMBDA,
                c_label="Annual exceedance rate (1/yr): RunUp >= threshold",
                title=f"{nameCase} — Annual exceedance rate map\nEQ RP = {eqrp} yr | Type = {mech_lbl} | Source = {src_lbl}",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

        # ---- Comparison: alongshore by source (same eqrp/type) for each T ----
        # This makes it easy to compare transects vs areas vs both.
        by_eqrp_type = {}
        for (eqrp, typ, src) in keys:
            if want_eqrps and eqrp not in want_eqrps:
                continue
            if want_types and typ not in want_types:
                continue
            if want_sources and src not in want_sources:
                continue
            by_eqrp_type.setdefault((eqrp, typ), []).append(src)

        for (eqrp, typ), srcs in sorted(by_eqrp_type.items()):
            srcs = sorted(set(srcs))
            mech_lbl = {"trans": "translational", "rot": "rotational", "any": "combined"}.get(typ, typ)

            for T in Ts:
                plt.figure()
                did_any = False
                for src in srcs:
                    fcsv = best[(eqrp, typ, src)][0]
                    df = pd.read_csv(fcsv)
                    tmp = idx[["transect_id", "coast_s_m"]].rename(columns={"transect_id": "coast_id"})
                    df["coast_id"] = pd.to_numeric(df["coast_id"], errors="coerce").astype("Int64")
                    df = df.merge(tmp, on="coast_id", how="left")
                    if not df["coast_s_m"].notna().any():
                        continue
                    df = df.sort_values("coast_s_m")
                    xa = pd.to_numeric(df["coast_s_m"], errors="coerce").to_numpy(float)
                    yv = pd.to_numeric(df[f"RunUp_{T}yr"], errors="coerce").to_numpy(float)
                    src_lbl = {"transects": "transects", "areas": "areas", "both": "transects+areas"}.get(src, src)
                    plt.plot(xa, yv, marker="o", linewidth=1.5, markersize=3, label=src_lbl)
                    did_any = True

                if not did_any:
                    plt.close()
                    continue

                plt.xlabel("Alongshore distance coast_s_m [m]")
                plt.ylabel(f"RunUp_{T}yr [m]")
                plt.title(f"{nameCase} — Compare sources alongshore\nEQ RP = {eqrp} yr | Type = {mech_lbl} | Tsunami T = {T} yr")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                out_png = os.path.join(runup_dir, f"runup_compare_sources_eqRP{eqrp}_{typ}_T{T}.png")
                plt.savefig(out_png, dpi=200)
                plt.close()
                print("[OK]", out_png)

    # ==========================================================
    # 2) FAILURE FIGURES
    # ==========================================================
    if not all(c in idx.columns for c in ("transect_id", "x0", "y0", "coast_s_m")):
        raise SystemExit("[FAIL] index.csv must contain transect_id, x0, y0, coast_s_m for failure plots.")

    for mech in fail_mechs:
        if mech not in ("trans", "rot"):
            continue

        mech_lbl = {"trans": "translational", "rot": "rotational"}.get(mech, mech)

        # --------------------------
        # 2a) TRANSECTS summaries
        # --------------------------
        tr_files = sorted(glob.glob(os.path.join(transects_dir, f"tr_*_{mech}_mcs.csv")))
        rx_tr = re.compile(r"tr_(\d+)_" + re.escape(mech) + r"_mcs\.csv$", re.IGNORECASE)

        rows_tr = []
        for fcsv in tr_files:
            m = rx_tr.search(os.path.basename(fcsv))
            if not m:
                continue
            tid = int(m.group(1))

            dfm = pd.read_csv(fcsv)
            st = failure_stats_from_df(dfm)
            if st["n"] <= 0:
                continue

            rows_tr.append({
                "transect_id": tid,
                "Pf_cap": st["Pf"],
                "V_p50_m3": pct_or_nan(st["Vfail"], 50),
                "V_p90_m3": pct_or_nan(st["Vfail"], 90),
                "A_p50_m2": pct_or_nan(st["Afail"], 50),
                "A_p90_m2": pct_or_nan(st["Afail"], 90),
                "n": st["n"],
                "nfail": st["nfail"],
            })

        S_tr = pd.DataFrame(rows_tr)
        if len(S_tr):
            S_tr["transect_id"] = pd.to_numeric(S_tr["transect_id"], errors="coerce").astype("Int64")
            S_tr = S_tr.merge(idx[["transect_id", "x0", "y0", "coast_s_m"]], on="transect_id", how="left")
            S_tr = S_tr.sort_values("coast_s_m").reset_index(drop=True)

            xa = pd.to_numeric(S_tr["coast_s_m"], errors="coerce").to_numpy(float)

            # Pf alongshore
            plt.figure()
            y = pd.to_numeric(S_tr["Pf_cap"], errors="coerce").to_numpy(float)
            plt.plot(xa, y, marker="o", linewidth=1.5, markersize=3)
            plt.xlabel("Alongshore distance coast_s_m [m]")
            plt.ylabel("Failure probability (within RP cap)")
            plt.title(f"{nameCase} — Pf alongshore\nMechanism = {mech_lbl} | Source = transects")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(failure_dir, f"fail_pf_alongshore_{mech}_transects.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print("[OK]", out_png)

            # Volume alongshore
            plt.figure()
            y50 = pd.to_numeric(S_tr["V_p50_m3"], errors="coerce").to_numpy(float)
            y90 = pd.to_numeric(S_tr["V_p90_m3"], errors="coerce").to_numpy(float)
            plt.plot(xa, y50, marker="o", linewidth=1.5, markersize=3, label="p50")
            plt.plot(xa, y90, marker="o", linewidth=1.5, markersize=3, label="p90")
            plt.xlabel("Alongshore distance coast_s_m [m]")
            plt.ylabel("Failed-slide volume [$m^3$]")
            plt.title(f"{nameCase} — Failed-slide volume alongshore\nMechanism = {mech_lbl} | Source = transects")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(failure_dir, f"fail_volume_alongshore_{mech}_transects.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print("[OK]", out_png)

            # Area alongshore
            plt.figure()
            y50 = pd.to_numeric(S_tr["A_p50_m2"], errors="coerce").to_numpy(float)
            y90 = pd.to_numeric(S_tr["A_p90_m2"], errors="coerce").to_numpy(float)
            plt.plot(xa, y50, marker="o", linewidth=1.5, markersize=3, label="p50")
            plt.plot(xa, y90, marker="o", linewidth=1.5, markersize=3, label="p90")
            plt.xlabel("Alongshore distance coast_s_m [m]")
            plt.ylabel("Failed-slide area [$m^2$]")
            plt.title(f"{nameCase} — Failed-slide area alongshore\nMechanism = {mech_lbl} | Source = transects")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(failure_dir, f"fail_area_alongshore_{mech}_transects.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print("[OK]", out_png)

            # Maps for transects
            X = pd.to_numeric(S_tr["x0"], errors="coerce").to_numpy(float)
            Y = pd.to_numeric(S_tr["y0"], errors="coerce").to_numpy(float)

            out_png = os.path.join(failure_dir, f"fail_pf_map_{mech}_transects.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_tr["Pf_cap"], errors="coerce").to_numpy(float),
                cmap=CMAP_PF,
                c_label="Pf_cap (fraction)",
                title=f"{nameCase} — Failure probability map\nMechanism = {mech_lbl} | Source = transects",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

            out_png = os.path.join(failure_dir, f"fail_volume_p50_map_{mech}_transects.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_tr["V_p50_m3"], errors="coerce").to_numpy(float),
                cmap=CMAP_VOL,
                c_label="Failed-slide volume p50 [$m^3$]",
                title=f"{nameCase} — Failed-slide volume (p50) map\nMechanism = {mech_lbl} | Source = transects",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

            out_png = os.path.join(failure_dir, f"fail_area_p50_map_{mech}_transects.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_tr["A_p50_m2"], errors="coerce").to_numpy(float),
                cmap=CMAP_AREA,
                c_label="Failed-slide area p50 [$m^2$]",
                title=f"{nameCase} — Failed-slide area (p50) map\nMechanism = {mech_lbl} | Source = transects",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

        # --------------------------
        # 2b) AREAS centers summaries (maps only)
        # --------------------------
        ar_files = sorted(glob.glob(os.path.join(areas_dir, f"ar_*_{mech}_mcs.csv")))
        rx_ar = re.compile(r"ar_(\d+)_" + re.escape(mech) + r"_mcs\.csv$", re.IGNORECASE)

        rows_ar_cent = []
        for fcsv in ar_files:
            if not rx_ar.search(os.path.basename(fcsv)):
                continue
            dfm = pd.read_csv(fcsv)

            # need center coords for map
            if ("x_center_m" not in dfm.columns) or ("y_center_m" not in dfm.columns):
                continue

            st = failure_stats_from_df(dfm)
            if st["n"] <= 0:
                continue

            xc = float(pd.to_numeric(dfm["x_center_m"], errors="coerce").median())
            yc = float(pd.to_numeric(dfm["y_center_m"], errors="coerce").median())
            if not (np.isfinite(xc) and np.isfinite(yc)):
                continue

            rows_ar_cent.append({
                "x": xc,
                "y": yc,
                "Pf_cap": st["Pf"],
                "V_p50_m3": pct_or_nan(st["Vfail"], 50),
                "A_p50_m2": pct_or_nan(st["Afail"], 50),
            })

        if len(rows_ar_cent):
            S_ac = pd.DataFrame(rows_ar_cent)
            X = pd.to_numeric(S_ac["x"], errors="coerce").to_numpy(float)
            Y = pd.to_numeric(S_ac["y"], errors="coerce").to_numpy(float)

            out_png = os.path.join(failure_dir, f"fail_pf_map_{mech}_areas_centers.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_ac["Pf_cap"], errors="coerce").to_numpy(float),
                cmap=CMAP_PF,
                c_label="Pf_cap (fraction)",
                title=f"{nameCase} — Failure probability map\nMechanism = {mech_lbl} | Source = areas (centers)",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

            out_png = os.path.join(failure_dir, f"fail_volume_p50_map_{mech}_areas_centers.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_ac["V_p50_m3"], errors="coerce").to_numpy(float),
                cmap=CMAP_VOL,
                c_label="Failed-slide volume p50 [$m^3$]",
                title=f"{nameCase} — Failed-slide volume (p50) map\nMechanism = {mech_lbl} | Source = areas (centers)",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

            out_png = os.path.join(failure_dir, f"fail_area_p50_map_{mech}_areas_centers.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_ac["A_p50_m2"], errors="coerce").to_numpy(float),
                cmap=CMAP_AREA,
                c_label="Failed-slide area p50 [$m^2$]",
                title=f"{nameCase} — Failed-slide area (p50) map\nMechanism = {mech_lbl} | Source = areas (centers)",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

        # --------------------------
        # 2c) AREAS -> COAST aggregation (alongshore + maps)
        # --------------------------
        acc = {}  # coast_id -> aggregations

        for fcsv in ar_files:
            if not rx_ar.search(os.path.basename(fcsv)):
                continue
            dfm = pd.read_csv(fcsv)

            need_cols = {"x_center_m", "y_center_m"}
            if not need_cols.issubset(dfm.columns):
                continue

            st = failure_stats_from_df(dfm)
            if st["n"] <= 0:
                continue

            xc = float(pd.to_numeric(dfm["x_center_m"], errors="coerce").median())
            yc = float(pd.to_numeric(dfm["y_center_m"], errors="coerce").median())
            if not (np.isfinite(xc) and np.isfinite(yc)):
                continue

            cid, cs = nearest_coast_id(xc, yc, x_coast, y_coast, coast_ids, coast_s)

            slot = acc.get(cid)
            if slot is None:
                slot = dict(coast_id=cid, coast_s_m=cs, x=float(x_coast[np.where(coast_ids == cid)[0][0]]),
                            y=float(y_coast[np.where(coast_ids == cid)[0][0]]),
                            n=0, nfail=0, Vparts=[], Aparts=[])
                acc[cid] = slot

            slot["n"] += int(st["n"])
            slot["nfail"] += int(st["nfail"])
            if st["Vfail"].size:
                slot["Vparts"].append(st["Vfail"].astype(np.float32))
            if st["Afail"].size:
                slot["Aparts"].append(st["Afail"].astype(np.float32))

        rows_atc = []
        for cid, slot in acc.items():
            if slot["Vparts"]:
                Vfail = np.concatenate(slot["Vparts"]).astype(float)
            else:
                Vfail = np.array([], float)
            if slot["Aparts"]:
                Afail = np.concatenate(slot["Aparts"]).astype(float)
            else:
                Afail = np.array([], float)

            Pf = float(slot["nfail"] / slot["n"]) if slot["n"] > 0 else np.nan
            rows_atc.append({
                "coast_id": cid,
                "coast_s_m": slot["coast_s_m"],
                "x0": slot["x"],
                "y0": slot["y"],
                "Pf_cap": Pf,
                "V_p50_m3": pct_or_nan(Vfail, 50),
                "V_p90_m3": pct_or_nan(Vfail, 90),
                "A_p50_m2": pct_or_nan(Afail, 50),
                "A_p90_m2": pct_or_nan(Afail, 90),
                "n": slot["n"],
                "nfail": slot["nfail"],
            })

        S_atc = pd.DataFrame(rows_atc)
        if len(S_atc):
            S_atc = S_atc.sort_values("coast_s_m").reset_index(drop=True)
            xa = pd.to_numeric(S_atc["coast_s_m"], errors="coerce").to_numpy(float)

            # Pf alongshore
            plt.figure()
            y = pd.to_numeric(S_atc["Pf_cap"], errors="coerce").to_numpy(float)
            plt.plot(xa, y, marker="s", linewidth=1.5, markersize=3)
            plt.xlabel("Alongshore distance coast_s_m [m]")
            plt.ylabel("Failure probability (within RP cap)")
            plt.title(f"{nameCase} — Pf alongshore\nMechanism = {mech_lbl} | Source = areas→coast")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(failure_dir, f"fail_pf_alongshore_{mech}_areas_to_coast.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print("[OK]", out_png)

            # Volume alongshore
            plt.figure()
            y50 = pd.to_numeric(S_atc["V_p50_m3"], errors="coerce").to_numpy(float)
            y90 = pd.to_numeric(S_atc["V_p90_m3"], errors="coerce").to_numpy(float)
            plt.plot(xa, y50, marker="s", linewidth=1.5, markersize=3, label="p50")
            plt.plot(xa, y90, marker="s", linewidth=1.5, markersize=3, label="p90")
            plt.xlabel("Alongshore distance coast_s_m [m]")
            plt.ylabel("Failed-slide volume [$m^3$]")
            plt.title(f"{nameCase} — Failed-slide volume alongshore\nMechanism = {mech_lbl} | Source = areas→coast")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(failure_dir, f"fail_volume_alongshore_{mech}_areas_to_coast.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print("[OK]", out_png)

            # Area alongshore
            plt.figure()
            y50 = pd.to_numeric(S_atc["A_p50_m2"], errors="coerce").to_numpy(float)
            y90 = pd.to_numeric(S_atc["A_p90_m2"], errors="coerce").to_numpy(float)
            plt.plot(xa, y50, marker="s", linewidth=1.5, markersize=3, label="p50")
            plt.plot(xa, y90, marker="s", linewidth=1.5, markersize=3, label="p90")
            plt.xlabel("Alongshore distance coast_s_m [m]")
            plt.ylabel("Failed-slide area [$m^2$]")
            plt.title(f"{nameCase} — Failed-slide area alongshore\nMechanism = {mech_lbl} | Source = areas→coast")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(failure_dir, f"fail_area_alongshore_{mech}_areas_to_coast.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print("[OK]", out_png)

            # Maps
            X = pd.to_numeric(S_atc["x0"], errors="coerce").to_numpy(float)
            Y = pd.to_numeric(S_atc["y0"], errors="coerce").to_numpy(float)

            out_png = os.path.join(failure_dir, f"fail_pf_map_{mech}_areas_to_coast.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_atc["Pf_cap"], errors="coerce").to_numpy(float),
                cmap=CMAP_PF,
                c_label="Pf_cap (fraction)",
                title=f"{nameCase} — Failure probability map\nMechanism = {mech_lbl} | Source = areas→coast",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

            out_png = os.path.join(failure_dir, f"fail_volume_p50_map_{mech}_areas_to_coast.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_atc["V_p50_m3"], errors="coerce").to_numpy(float),
                cmap=CMAP_VOL,
                c_label="Failed-slide volume p50 [$m^3$]",
                title=f"{nameCase} — Failed-slide volume (p50) map\nMechanism = {mech_lbl} | Source = areas→coast",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

            out_png = os.path.join(failure_dir, f"fail_area_p50_map_{mech}_areas_to_coast.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, pd.to_numeric(S_atc["A_p50_m2"], errors="coerce").to_numpy(float),
                cmap=CMAP_AREA,
                c_label="Failed-slide area p50 [$m^2$]",
                title=f"{nameCase} — Failed-slide area (p50) map\nMechanism = {mech_lbl} | Source = areas→coast",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

        # --------------------------
        # 2d) Compare volume alongshore: transects vs areas→coast
        # --------------------------
        if len(S_tr) and len(S_atc):
            plt.figure()
            plt.plot(S_tr["coast_s_m"], S_tr["V_p50_m3"], marker="o", linewidth=1.5, markersize=3, label="Transects V p50")
            plt.plot(S_tr["coast_s_m"], S_tr["V_p90_m3"], marker="o", linewidth=1.5, markersize=3, label="Transects V p90")
            plt.plot(S_atc["coast_s_m"], S_atc["V_p50_m3"], marker="s", linewidth=1.5, markersize=3, label="Areas→coast V p50")
            plt.plot(S_atc["coast_s_m"], S_atc["V_p90_m3"], marker="s", linewidth=1.5, markersize=3, label="Areas→coast V p90")
            plt.xlabel("Alongshore distance coast_s_m [m]")
            plt.ylabel("Failed-slide volume [$m^3$]")
            plt.title(f"{nameCase} — Compare failed-slide volumes alongshore\nMechanism = {mech_lbl} | Transects vs Areas→coast")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out_png = os.path.join(failure_dir, f"compare_volume_alongshore_{mech}.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print("[OK]", out_png)

    print("[DONE] figures in:")
    print("  ", runup_dir)
    print("  ", failure_dir)


if __name__ == "__main__":
    main()