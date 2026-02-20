#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# project root on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utilities.config import load_config, abspath


# -----------------------------
# Plot helpers
# -----------------------------
def load_bathy(depth_tif):
    with rasterio.open(depth_tif) as src:
        Z = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None and not (isinstance(nod, float) and np.isnan(nod)):
            Z[Z == nod] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    return Z, extent


def plot_points_on_bathy(
    Z, extent, x, y, values, title, outpng,
    label_ids=None, label_x=None, label_y=None,
    cbar_label=None,
    bathy_cmap="viridis",
    values_cmap="jet",
    cbar_fraction=0.045,
    cbar_pad=0.02,
    cbar_shrink=0.85,
    tick_labelsize=8,
):
    os.makedirs(os.path.dirname(outpng) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Background bathy (viridis)
    im = ax.imshow(
        np.ma.masked_invalid(Z),
        extent=extent,
        origin="upper",
        cmap=bathy_cmap,
    )

    # Bathy colorbar: horizontal + smaller
    cbar_b = fig.colorbar(
        im, ax=ax,
        orientation="horizontal",
        fraction=cbar_fraction,
        pad=0.06,
        shrink=cbar_shrink,
    )
    cbar_b.set_label("bathy (m)")
    cbar_b.ax.tick_params(labelsize=tick_labelsize)

    # Points (jet)
    sc = ax.scatter(
        x, y,
        c=values,
        s=60,
        cmap=values_cmap,
        edgecolor="k",
        linewidth=0.3,
    )

    # Values colorbar: vertical + smaller
    cbar_s = fig.colorbar(
        sc, ax=ax,
        orientation="vertical",
        fraction=cbar_fraction,
        pad=cbar_pad,
        shrink=cbar_shrink,
    )
    cbar_s.set_label(cbar_label if cbar_label else title)
    cbar_s.ax.tick_params(labelsize=tick_labelsize)

    # Labels
    if label_ids is not None and label_x is not None and label_y is not None:
        for tid, xx, yy in zip(label_ids, label_x, label_y):
            if not (np.isfinite(xx) and np.isfinite(yy)):
                continue
            ax.text(
                xx, yy, str(int(tid)),
                fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
            )

    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    fig.tight_layout()
    fig.savefig(outpng, dpi=250)
    plt.close(fig)
    print("[OK] wrote", outpng)


def pick_label_points(df, label_top_n, by_col, id_col="transect_id", x_col="x0", y_col="y0"):
    if (not label_top_n) or label_top_n <= 0:
        return None, None, None
    if by_col not in df.columns:
        return None, None, None

    tmp = df[[id_col, x_col, y_col, by_col]].copy()
    tmp = tmp[np.isfinite(tmp[by_col].to_numpy(float))]
    if tmp.empty:
        return None, None, None

    top = tmp.sort_values(by_col, ascending=False).head(int(label_top_n))
    return top[id_col].to_numpy(), top[x_col].to_numpy(), top[y_col].to_numpy()


def ensure_xy(summary_df, index_csv):
    """
    Ensure summary has x0,y0. If missing, merge from index.csv.
    """
    have = set(summary_df.columns)
    if ("x0" in have) and ("y0" in have):
        return summary_df

    if not os.path.exists(index_csv):
        raise SystemExit(f"summary missing x0/y0 and index.csv not found: {index_csv}")

    idx = pd.read_csv(index_csv)
    if "id" in idx.columns and "transect_id" not in idx.columns:
        idx = idx.rename(columns={"id": "transect_id"})
    need = {"transect_id", "x0", "y0"}
    if not need.issubset(idx.columns):
        raise SystemExit(f"index.csv missing {need}, has {list(idx.columns)}")

    out = summary_df.merge(idx[["transect_id", "x0", "y0"]], on="transect_id", how="left")
    return out


def _sort_rps(rps):
    def key(r):
        try:
            return int(str(r))
        except Exception:
            return 10**18
    return sorted([str(r) for r in rps], key=key)


def _existing_rps_in_summary(df):
    # infer RPs from columns like Pf_475_any or eta0_fail475_p99_m_any
    rps = set()
    for c in df.columns:
        if c.startswith("Pf_"):
            parts = c.split("_")
            if len(parts) >= 3:
                rps.add(parts[1])
        elif c.startswith("eta0_fail"):
            # eta0_fail{rp}_pXX_m_{typ}
            s = c[len("eta0_fail"):]
            # take leading digits as rp
            rp = ""
            for ch in s:
                if ch.isdigit():
                    rp += ch
                else:
                    break
            if rp:
                rps.add(rp)
    return _sort_rps(rps)


def choose_default_label_col(df, rps, types, eta_pcts, prefer_type="any"):
    """
    Priority:
      1) eta0_fail{max_rp}_p{max_pct}_m_{prefer_type}
      2) eta0_fail{max_rp}_p{max_pct}_m_{first_available_type}
      3) Pf_{max_rp}_{prefer_type}
      4) Pf_{max_rp}_{first_available_type}
      5) first eta0_fail... column
      6) first Pf_... column
    """
    rps = _sort_rps(rps)
    if not rps:
        return None
    rp = rps[-1]
    eta_pcts = sorted([int(p) for p in eta_pcts]) if eta_pcts else [99]
    p = eta_pcts[-1]

    types_in = [t for t in types if t] if types else ["any", "trans", "rot"]
    types_pref = [prefer_type] + [t for t in types_in if t != prefer_type]

    for typ in types_pref:
        c = f"eta0_fail{rp}_p{p:02d}_m_{typ}"
        if c in df.columns:
            return c

    for typ in types_pref:
        c = f"Pf_{rp}_{typ}"
        if c in df.columns:
            return c

    for c in df.columns:
        if c.startswith("eta0_fail"):
            return c
    for c in df.columns:
        if c.startswith("Pf_"):
            return c

    return None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 6: Make Pf/eta0 point maps from summary CSV (config-driven).")
    ap.add_argument("--config", required=True, help="Path to case config.json")

    ap.add_argument("--summary-csv", default=None,
                    help="Override summary CSV (else config outputs.summary_csv)")
    ap.add_argument("--index-csv", default=None,
                    help="Override index.csv (else <transects_dir>/index.csv)")
    ap.add_argument("--depth-tif", default=None,
                    help="Override bathy tif (else config paths.bathy_tif)")
    ap.add_argument("--out-dir", default=None,
                    help="Override output maps dir (else config outputs.maps_dir)")

    ap.add_argument("--rps", default=None,
                    help="Comma-separated return periods to map. "
                         "Default: infer from summary columns.")
    ap.add_argument("--types", default="any,trans,rot",
                    help="Comma-separated types to map: any,trans,rot")

    ap.add_argument("--eta-pcts", default="99",
                    help="Comma-separated percentiles for eta0_fail (e.g. '90,95,99')")
    ap.add_argument("--label-top-n", type=int, default=10,
                    help="Label top-N transects on maps (0 disables).")

    ap.add_argument("--label-by", default=None,
                    help="Column to rank for labels. Default: auto-pick (highest RP + highest eta pct + type=any if possible).")

    args = ap.parse_args()

    cfg = load_config(args.config)

    transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"])
    summary_csv = abspath(cfg, cfg["outputs"]["summary_csv"]) if args.summary_csv is None else args.summary_csv
    index_csv = os.path.join(transects_dir, "index.csv") if args.index_csv is None else args.index_csv
    depth_tif = abspath(cfg, cfg["paths"]["bathy_tif"]) if args.depth_tif is None else args.depth_tif
    out_dir = abspath(cfg, cfg["outputs"].get("maps_dir", os.path.join(transects_dir, "maps"))) if args.out_dir is None else args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(summary_csv):
        raise SystemExit(f"Missing summary CSV: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if "transect_id" not in df.columns:
        raise SystemExit("summary CSV missing 'transect_id' column.")

    df["transect_id"] = df["transect_id"].astype(int)
    df = ensure_xy(df, index_csv)

    if df[["x0", "y0"]].isna().any().any():
        print("[WARN] Some x0/y0 are NaN after merge. Those points may not plot correctly.")

    types = [s.strip() for s in str(args.types).split(",") if s.strip() != ""]
    eta_pcts = [int(s.strip()) for s in str(args.eta_pcts).split(",") if s.strip() != ""]

    # RPs: either user-specified or inferred from summary
    if args.rps:
        rps = [s.strip() for s in str(args.rps).split(",") if s.strip() != ""]
        rps = _sort_rps(rps)
    else:
        rps = _existing_rps_in_summary(df)

    if not rps:
        raise SystemExit("Could not infer any RPs from summary, and --rps not provided.")

    # Background
    if not os.path.exists(depth_tif):
        raise SystemExit(f"Missing depth/bathy tif: {depth_tif}")
    Zb, extent = load_bathy(depth_tif)

    x = df["x0"].to_numpy(float)
    y = df["y0"].to_numpy(float)

    # Label selection
    label_col = args.label_by
    if label_col is None:
        label_col = choose_default_label_col(df, rps=rps, types=types, eta_pcts=eta_pcts, prefer_type="any")

    label_ids = label_x = label_y = None
    if label_col is not None:
        label_ids, label_x, label_y = pick_label_points(df, args.label_top_n, label_col)
        if label_ids is not None:
            print(f"[INFO] labeling top {args.label_top_n} by: {label_col}")
    else:
        print("[INFO] labeling disabled (no suitable label column found).")
        label_ids = label_x = label_y = None

    mapped_cols = []

    # -----------------------------
    # MAPS: Pf and eta0_fail percentiles per RP/type
    # -----------------------------
    for rp in rps:
        for typ in types:
            pf_col = f"Pf_{rp}_{typ}"
            if pf_col in df.columns:
                outpng = os.path.join(out_dir, f"map_{pf_col}.png")
                plot_points_on_bathy(
                    Zb, extent, x, y, df[pf_col].to_numpy(float),
                    title=pf_col,
                    outpng=outpng,
                    label_ids=label_ids, label_x=label_x, label_y=label_y,
                    cbar_label=pf_col
                )
                mapped_cols.append(pf_col)
            else:
                print("[SKIP] missing column:", pf_col)

            for p in eta_pcts:
                eta_col = f"eta0_fail{rp}_p{p:02d}_m_{typ}"
                if eta_col in df.columns:
                    outpng = os.path.join(out_dir, f"map_{eta_col}.png")
                    plot_points_on_bathy(
                        Zb, extent, x, y, df[eta_col].to_numpy(float),
                        title=eta_col,
                        outpng=outpng,
                        label_ids=label_ids, label_x=label_x, label_y=label_y,
                        cbar_label=eta_col
                    )
                    mapped_cols.append(eta_col)
                else:
                    print("[SKIP] missing column:", eta_col)

    inv = pd.DataFrame({"mapped_column": mapped_cols})
    inv_path = os.path.join(out_dir, "maps_inventory.csv")
    inv.to_csv(inv_path, index=False)
    print("\n[DONE] maps in:", out_dir)
    print("[DONE] inventory:", inv_path)


if __name__ == "__main__":
    main()
