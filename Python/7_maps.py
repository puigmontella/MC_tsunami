#!/usr/bin/env python3
"""
7_maps.py

Purpose
-------
Create figures (alongshore curves + maps) from the Monte Carlo pipeline outputs.

Inputs
------
Background raster:
  - <rasters_dir>/depth_pos.tif        (positive water depth in meters)

Runup results (from step "coast runup"):
  - <coast_dir>/coast_runup_eqRP*_*.csv

Failure results (from step "MC"):
  - <transects_dir>/tr_*_trans_mcs.csv
  - <transects_dir>/tr_*_rot_mcs.csv

Transect/coast index (geometry + alongshore distance):
  - <transects_dir>/index.csv          (must include: transect_id, x0, y0, coast_s_m)

Outputs
-------
Figures in:
  - <maps_dir>/*.png

Notes
-----
- We plot points in the raster CRS (often meters), BUT format axis tick labels in lon/lat
  degrees+minutes by transforming tick positions to EPSG:4326.
- Alongshore curves always use coast_s_m (no fallback to transect_id).
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
    """
    Convert decimal degrees to a compact degrees+minutes label.

    Examples:
      lon: -16.50 -> 16°30'W
      lat:  28.45 -> 28°27'N
    """
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
    """
    Make one map:
      - background: depth raster (imshow)
      - overlay: points (scatter) colored by c
      - 2 colorbars: one for depth, one for c

    Axis ticks:
      We keep coordinates in raster CRS (x/y), but format tick labels as lon/lat
      degrees+minutes by transforming tick positions to EPSG:4326.
    """
    # Ensure numeric arrays
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.asarray(c, float)

    # Valid points mask
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if not np.any(m):
        return False

    # Color limits for the point data: use min/max (no robust clipping)
    cmin = float(np.nanmin(c[m]))
    cmax = float(np.nanmax(c[m]))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
        cmin, cmax = 0.0, 1.0

    # Create figure
    fig, ax = plt.subplots()

    # --- background raster ---
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

    # --- overlay points ---
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

    # --- axis label formatting (lon/lat deg+min) ---
    # We transform *tick coordinates* (in raster CRS) to EPSG:4326 for labels.
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

        # Make x tick labels readable
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
            lab.set_ha("right")
    else:
        # If CRS is unknown, just label in native x/y
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # --- two colorbars using axes_grid1 ---
    divider = make_axes_locatable(ax)

    # Depth colorbar at bottom (horizontal)
    cax_depth = divider.append_axes("bottom", size="6%", pad=0.55)
    cb_depth = fig.colorbar(im, cax=cax_depth, orientation="horizontal")
    cb_depth.set_label("Depth [m]")

    # Data colorbar at right (vertical)
    cax_data = divider.append_axes("right", size="4%", pad=0.15)
    cb_data = fig.colorbar(sc, cax=cax_data)
    cb_data.set_label(c_label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


# -----------------------------
# Main
# -----------------------------
def main():
    # ---- CLI arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--types", default=None,
                    help="Comma list: trans,rot,any. Default: all available.")
    ap.add_argument("--eq-rps", default=None,
                    help="Comma list: 100,475,2475,... Default: all available.")
    ap.add_argument("--map-Ts", default=None,
                    help="Comma list: 100,200,2475. Default: 100,200,2475.")
    args = ap.parse_args()

    # ---- Read project config ----
    with open(args.config, "r") as f:
        cfg = json.load(f)

    transects_dir = cfg["outputs"]["transects_dir"]
    coast_dir = cfg["outputs"]["coast_dir"]
    maps_dir = cfg["outputs"]["maps_dir"]
    rasters_dir = cfg["outputs"]["rasters_dir"]
    os.makedirs(maps_dir, exist_ok=True)

    # Use current folder name as "case name" in plot titles
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
        extent = (b.left, b.right, b.bottom, b.top)  # (left, right, bottom, top)
        src_crs = ds.crs  # needed to format axes as lon/lat

    # Depth color scaling (simple min/max)
    dm = np.isfinite(depth)
    depth_vmin = float(np.nanmin(depth[dm])) if np.any(dm) else None
    depth_vmax = float(np.nanmax(depth[dm])) if np.any(dm) else None

    # ==========================================================
    # B) Load index.csv (coast geometry + coast_s_m)
    # ==========================================================
    idx_path = os.path.join(transects_dir, "index.csv")
    if not os.path.exists(idx_path):
        raise SystemExit(f"[FAIL] Missing index.csv: {idx_path}")

    idx = pd.read_csv(idx_path)
    if "id" in idx.columns:
        idx = idx.rename(columns={"id": "transect_id"})
    idx["transect_id"] = pd.to_numeric(idx["transect_id"], errors="coerce").astype("Int64")

    # coast_s_m is mandatory (no fallback)
    if "coast_s_m" not in idx.columns:
        raise SystemExit("[FAIL] index.csv must contain 'coast_s_m' for alongshore plots.")

    # ==========================================================
    # C) Parse filters: mechanism types and earthquake RPs
    # ==========================================================
    want_types = None
    if args.types:
        want_types = set(t.strip().lower() for t in args.types.split(",") if t.strip())
        if "any" in want_types:
            want_types = None  # "any" => don't filter by type

    want_eqrps = None
    if args.eq_rps:
        want_eqrps = set(int(x.strip()) for x in args.eq_rps.split(",") if x.strip())

    # Tsunami design return periods to plot on maps/curves
    Ts = [100, 500, 2475]
    if args.map_Ts:
        Ts = [int(x.strip()) for x in args.map_Ts.split(",") if x.strip()]

    # ==========================================================
    # 1) RUNUP FIGURES
    # ==========================================================
    # Files come from the coast-runup step:
    #   coast_runup_eqRP{eqrp}_{type}.csv
    coast_files = sorted(glob.glob(os.path.join(coast_dir, "coast_runup_eqRP*_*.csv")))
    if not coast_files:
        print(f"[WARN] No coast_runup_eqRP*_*.csv found in: {coast_dir}")

    for fcsv in coast_files:
        base = os.path.basename(fcsv)

        # Parse eqrp + mechanism from filename
        m = re.search(r"coast_runup_eqRP(\d+)_([a-zA-Z]+)\.csv$", base)
        if not m:
            continue
        eqrp = int(m.group(1))
        mech = m.group(2).lower()

        # Apply CLI filters
        if want_types and mech not in want_types:
            continue
        if want_eqrps and eqrp not in want_eqrps:
            continue

        # Label used in titles
        mech_lbl = {"trans": "translational", "rot": "rotational", "any": "combined"}.get(mech, mech)

        # Read runup table
        df = pd.read_csv(fcsv)

        # Basic sanity check for columns needed downstream
        required = {"coast_id", "x", "y", "lambda_RunUp_ge_thr_1yr"}
        for T in Ts:
            required.add(f"RunUp_{T}yr")
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] skipping {base} (missing columns: {missing})")
            continue

        # Make sure key fields are numeric
        df["coast_id"] = pd.to_numeric(df["coast_id"], errors="coerce").astype("Int64")
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        # Attach coast_s_m using index.csv (coast_id must match transect_id)
        tmp = idx[["transect_id", "coast_s_m"]].rename(columns={"transect_id": "coast_id"})
        df = df.merge(tmp, on="coast_id", how="left")
        if not df["coast_s_m"].notna().any():
            raise SystemExit(f"[FAIL] Could not map coast_s_m onto {base}. Check coast_id alignment with index.csv.")

        # Sort by alongshore distance
        df = df.sort_values("coast_s_m").reset_index(drop=True)

        # --------------------------
        # 1a) Alongshore runup curves
        # --------------------------
        plt.figure()
        xa = pd.to_numeric(df["coast_s_m"], errors="coerce").to_numpy(float)
        for T in Ts:
            yv = pd.to_numeric(df[f"RunUp_{T}yr"], errors="coerce").to_numpy(float)
            plt.plot(xa, yv, marker="o", linewidth=1.5, markersize=3,
                     label=f"Tsunami design level: RunUp_{T}yr [m]")

        plt.xlabel("Alongshore distance coast_s_m [m]")
        plt.ylabel("Runup [m]")
        plt.title(f"{nameCase} — Design runup alongshore\nEQ RP = {eqrp} yr \nMechanism = {mech_lbl}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(maps_dir, f"runup_alongshore_eqRP{eqrp}_{mech}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("[OK]", out_png)

        # --------------------------
        # 1b) Runup maps (per T)
        # --------------------------
        X = df["x"].to_numpy(float)
        Y = df["y"].to_numpy(float)

        for T in Ts:
            ccol = f"RunUp_{T}yr"
            C = pd.to_numeric(df[ccol], errors="coerce").to_numpy(float)

            out_png = os.path.join(maps_dir, f"runup_map_eqRP{eqrp}_{mech}_T{T}.png")
            ok = scatter_on_depth(
                depth, extent, X, Y, C,
                cmap=CMAP_RUNUP,
                c_label=f"{ccol} [m]",
                title=f"{nameCase} — Runup map\nEQ RP = {eqrp} yr \nMechanism = {mech_lbl} — Tsunami T = {T} yr",
                out_png=out_png,
                src_crs=src_crs,
                depth_vmin=depth_vmin, depth_vmax=depth_vmax
            )
            if ok:
                print("[OK]", out_png)

        # --------------------------
        # 1c) Annual exceedance rate map
        # --------------------------
        C = pd.to_numeric(df["lambda_RunUp_ge_thr_1yr"], errors="coerce").to_numpy(float)
        out_png = os.path.join(maps_dir, f"lambda_map_eqRP{eqrp}_{mech}.png")
        ok = scatter_on_depth(
            depth, extent, X, Y, C,
            cmap=CMAP_LAMBDA,
            c_label="Annual exceedance rate (1/yr): RunUp >= threshold",
            title=f"{nameCase} — Annual exceedance rate map\nEQ RP = {eqrp} yr \nMechanism = {mech_lbl}",
            out_png=out_png,
            src_crs=src_crs,
            depth_vmin=depth_vmin, depth_vmax=depth_vmax
        )
        if ok:
            print("[OK]", out_png)

    # ==========================================================
    # 2) FAILURE FIGURES (Pf + volume/area summaries)
    # ==========================================================
    # We need x0/y0 for maps and coast_s_m for alongshore distance
    if not all(c in idx.columns for c in ("transect_id", "x0", "y0", "coast_s_m")):
        raise SystemExit("[FAIL] index.csv must contain transect_id, x0, y0, coast_s_m for failure plots.")

    # MC output files produced by your MC scripts
    mcs_files = sorted(
        glob.glob(os.path.join(transects_dir, "tr_*_trans_mcs.csv"))
        + glob.glob(os.path.join(transects_dir, "tr_*_rot_mcs.csv"))
    )
    if not mcs_files:
        print(f"[WARN] No MC files found: {transects_dir}/tr_*_(trans|rot)_mcs.csv")
        print("[DONE] figures in:", maps_dir)
        return

    rx = re.compile(r"tr_(\d+)_(trans|rot)_mcs\.csv$", re.IGNORECASE)

    # Store per-transect summary rows per mechanism
    summaries = {"trans": [], "rot": []}

    # --------------------------
    # 2a) Build per-transect summaries
    # --------------------------
    for fcsv in mcs_files:
        base = os.path.basename(fcsv)
        m = rx.match(base)
        if not m:
            continue

        tid = int(m.group(1))
        mech = m.group(2).lower()

        if want_types and mech not in want_types:
            continue

        dfm = pd.read_csv(fcsv)

        # Your failure summary logic requires these columns; otherwise skip
        need = {"L_eff_m", "w_m", "T_m", "failed_within_cap"}
        if not need.issubset(dfm.columns):
            continue

        # Extract + clean numeric arrays
        L = pd.to_numeric(dfm["L_eff_m"], errors="coerce").to_numpy(float)
        W = pd.to_numeric(dfm["w_m"], errors="coerce").to_numpy(float)
        T = pd.to_numeric(dfm["T_m"], errors="coerce").to_numpy(float)
        F = pd.to_numeric(dfm["failed_within_cap"], errors="coerce").to_numpy(float)

        ok = np.isfinite(L) & np.isfinite(W) & np.isfinite(T) & np.isfinite(F)
        if not np.any(ok):
            continue

        L = L[ok]; W = W[ok]; T = T[ok]; F = F[ok]

        # Simple geometry proxies
        A = L * W
        V = A * T

        # Pf within the RP cap = fraction of accepted trials with failed_within_cap == 1
        n = float(len(F))
        pf = float(np.sum(F >= 0.5) / n) if n > 0 else np.nan

        # Stats among failed trials only
        fail = (F >= 0.5) & np.isfinite(V) & np.isfinite(A)
        if np.any(fail):
            V_p50 = float(np.nanpercentile(V[fail], 50))
            V_p90 = float(np.nanpercentile(V[fail], 90))
            A_p50 = float(np.nanpercentile(A[fail], 50))
            A_p90 = float(np.nanpercentile(A[fail], 90))
        else:
            V_p50 = np.nan; V_p90 = np.nan
            A_p50 = np.nan; A_p90 = np.nan

        summaries[mech].append({
            "transect_id": tid,
            "Pf_cap": pf,
            "V_p50_m3": V_p50,
            "V_p90_m3": V_p90,
            "A_p50_m2": A_p50,
            "A_p90_m2": A_p90,
        })

    # --------------------------
    # 2b) Plot alongshore curves + maps for each mechanism
    # --------------------------
    for mech, rows in summaries.items():
        if not rows:
            continue

        mech_lbl = {"trans": "translational", "rot": "rotational"}.get(mech, mech)

        S = pd.DataFrame(rows)
        S["transect_id"] = pd.to_numeric(S["transect_id"], errors="coerce").astype("Int64")

        # Attach x0/y0 and coast_s_m from index.csv
        S = S.merge(idx[["transect_id", "x0", "y0", "coast_s_m"]],
                    on="transect_id", how="left")

        if not S["coast_s_m"].notna().any():
            raise SystemExit(f"[FAIL] Missing coast_s_m after merge for mechanism={mech}.")

        # Sort alongshore
        S = S.sort_values("coast_s_m").reset_index(drop=True)
        xa = pd.to_numeric(S["coast_s_m"], errors="coerce").to_numpy(float)

        # ---- Pf alongshore ----
        plt.figure()
        y = pd.to_numeric(S["Pf_cap"], errors="coerce").to_numpy(float)
        plt.plot(xa, y, marker="o", linewidth=1.5, markersize=3)
        plt.xlabel("Alongshore distance [m]")
        plt.ylabel("Failure probability")
        plt.title(f"{nameCase} — Failure probability alongshore\nMechanism = {mech_lbl}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(maps_dir, f"fail_pf_alongshore_{mech}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("[OK]", out_png)

        # ---- Volume alongshore (p50 + p90) ----
        plt.figure()
        y50 = pd.to_numeric(S["V_p50_m3"], errors="coerce").to_numpy(float)
        y90 = pd.to_numeric(S["V_p90_m3"], errors="coerce").to_numpy(float)
        plt.plot(xa, y50, marker="o", linewidth=1.5, markersize=3, label="p50")
        plt.plot(xa, y90, marker="o", linewidth=1.5, markersize=3, label="p90")
        plt.xlabel("Alongshore distance [m]")
        plt.ylabel("Volume [$m^3$]")
        plt.title(f"{nameCase} — Failed-slide volume alongshore\nMechanism = {mech_lbl}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(maps_dir, f"fail_volume_alongshore_{mech}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("[OK]", out_png)

        # ---- Area alongshore (p50 + p90) ----
        plt.figure()
        y50 = pd.to_numeric(S["A_p50_m2"], errors="coerce").to_numpy(float)
        y90 = pd.to_numeric(S["A_p90_m2"], errors="coerce").to_numpy(float)
        plt.plot(xa, y50, marker="o", linewidth=1.5, markersize=3, label="p50")
        plt.plot(xa, y90, marker="o", linewidth=1.5, markersize=3, label="p90")
        plt.xlabel("Alongshore distance [m]")
        plt.ylabel("Area [$m^2$]")
        plt.title(f"{nameCase} — Failed-slide area alongshore\nMechanism = {mech_lbl}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(maps_dir, f"fail_area_alongshore_{mech}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("[OK]", out_png)

        # ---- Maps for Pf / volume / area ----
        X = pd.to_numeric(S["x0"], errors="coerce").to_numpy(float)
        Y = pd.to_numeric(S["y0"], errors="coerce").to_numpy(float)

        out_png = os.path.join(maps_dir, f"fail_pf_map_{mech}.png")
        ok = scatter_on_depth(
            depth, extent, X, Y, pd.to_numeric(S["Pf_cap"], errors="coerce").to_numpy(float),
            cmap=CMAP_PF,
            c_label="Pf_cap (fraction)",
            title=f"{nameCase} — Failure probability map\nMechanism = {mech_lbl}",
            out_png=out_png,
            src_crs=src_crs,
            depth_vmin=depth_vmin, depth_vmax=depth_vmax
        )
        if ok:
            print("[OK]", out_png)

        out_png = os.path.join(maps_dir, f"fail_volume_p50_map_{mech}.png")
        ok = scatter_on_depth(
            depth, extent, X, Y, pd.to_numeric(S["V_p50_m3"], errors="coerce").to_numpy(float),
            cmap=CMAP_VOL,
            c_label="Failed-slide volume p50 [$m^3$]",
            title=f"{nameCase} — Failed-slide volume (p50) map\nMechanism = {mech_lbl}",
            out_png=out_png,
            src_crs=src_crs,
            depth_vmin=depth_vmin, depth_vmax=depth_vmax
        )
        if ok:
            print("[OK]", out_png)

        out_png = os.path.join(maps_dir, f"fail_area_p50_map_{mech}.png")
        ok = scatter_on_depth(
            depth, extent, X, Y, pd.to_numeric(S["A_p50_m2"], errors="coerce").to_numpy(float),
            cmap=CMAP_AREA,
            c_label="Failed-slide area p50 [$m^2$]",
            title=f"{nameCase} — Failed-slide area (p50) map\nMechanism = {mech_lbl}",
            out_png=out_png,
            src_crs=src_crs,
            depth_vmin=depth_vmin, depth_vmax=depth_vmax
        )
        if ok:
            print("[OK]", out_png)

    print("[DONE] figures in:", maps_dir)


if __name__ == "__main__":
    main()