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



import re

def eta_tag_cm(H_m: float) -> str:
    cm = int(round(float(H_m) * 100.0))
    return f"{cm:03d}cm"  # 0.5m -> 050cm

def infer_haz_tags_from_df(df):
    """
    Look for hazard columns like AERH050cm_any or RPH050cm_any and return tags: ["050cm", ...]
    """
    tags = set()
    for c in df.columns:
        m = re.match(r"^(AERH|RPH)(\d+cm)_(any|trans|rot)$", c)
        if m:
            tags.add(m.group(2))
    def key(t):
        try:
            return int(t.replace("cm", ""))
        except Exception:
            return 10**18
    return sorted(list(tags), key=key)

def pick_label_points(
    df, label_top_n, by_col,
    id_col="transect_id", x_col="x0", y_col="y0",
    ascending=False,
):
    if (not label_top_n) or label_top_n <= 0:
        return None, None, None
    if by_col not in df.columns:
        return None, None, None

    tmp = df[[id_col, x_col, y_col, by_col]].copy()
    tmp = tmp[np.isfinite(tmp[by_col].to_numpy(float))]
    if tmp.empty:
        return None, None, None

    top = tmp.sort_values(by_col, ascending=bool(ascending)).head(int(label_top_n))
    return top[id_col].to_numpy(), top[x_col].to_numpy(), top[y_col].to_numpy()
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

    # Background bathy
    im = ax.imshow(
        np.ma.masked_invalid(Z),
        extent=extent,
        origin="upper",
        cmap=bathy_cmap,
    )

    # Bathy colorbar
    cbar_b = fig.colorbar(
        im, ax=ax,
        orientation="horizontal",
        fraction=cbar_fraction,
        pad=0.06,
        shrink=cbar_shrink,
    )
    cbar_b.set_label("bathy (m)")
    cbar_b.ax.tick_params(labelsize=tick_labelsize)

    # Points
    sc = ax.scatter(
        x, y,
        c=values,
        s=60,
        cmap=values_cmap,
        edgecolor="k",
        linewidth=0.3,
    )

    # Values colorbar
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
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(outpng, dpi=250)
    plt.close(fig)
    print("[OK] wrote", outpng)




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
    """
    infer RPs from columns:
      Pf_475_any
      PfT_475_any
      eta0_fail475_p99_m_any
      eta0_tsu475_p99_m_any
      PSMF_475_any
      RP_SMF_475_any
    """
    rps = set()
    for c in df.columns:
        if c.startswith("Pf_") or c.startswith("PfT_") or c.startswith("PSMF_") or c.startswith("RP_SMF_"):
            parts = c.split("_")
            if len(parts) >= 3:
                rps.add(parts[1])

        elif c.startswith("eta0_fail"):
            s = c[len("eta0_fail"):]
            rp = ""
            for ch in s:
                if ch.isdigit():
                    rp += ch
                else:
                    break
            if rp:
                rps.add(rp)

        elif c.startswith("eta0_tsu"):
            s = c[len("eta0_tsu"):]
            rp = ""
            for ch in s:
                if ch.isdigit():
                    rp += ch
                else:
                    break
            if rp:
                rps.add(rp)

    return _sort_rps(rps)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 6: Make point maps from summary CSV (config-driven).")
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
                    help="Comma-separated return periods to map. Default: infer from summary columns.")
    ap.add_argument("--types", default="any,trans,rot",
                    help="Comma-separated types to map: any,trans,rot")

    ap.add_argument("--eta-pcts", default="99",
                    help="Comma-separated percentiles for eta0_* (e.g. '90,95,99')")

    ap.add_argument("--map-set", default="fail",
                    help="What to map: comma list among {fail,tsu,smf}. "
                         "fail=Pf+eta0_fail, tsu=PfT+eta0_tsu, smf=PSMF+RP_SMF. Example: --map-set fail,tsu")

    ap.add_argument("--label-top-n", type=int, default=10,
                    help="Label top-N transects on maps (0 disables).")

    ap.add_argument("--label-by", default=None,
                    help="GLOBAL labeling: one column to rank for labels across ALL maps. "
                         "If not set, default behavior is per-map labeling (labels match each map).")

    ap.add_argument("--label-mode", default=None, choices=["per-map", "global"],
                    help="Label mode. Default: per-map if --label-by not set, else global.")

    ap.add_argument("--haz-thr", default=None,
                    help="Eta thresholds in meters for hazard maps (comma list). "
                        "Example: --haz-thr 0.5 or --haz-thr 0.1,0.5,1.0")

    ap.add_argument("--haz-what", default="aer,rp",
                    help="Which hazard metrics to map: aer (AERH*) and/or rp (RPH*). Default: aer,rp")

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
    map_set = {s.strip().lower() for s in str(args.map_set).split(",") if s.strip() != ""}



    haz_what = {s.strip().lower() for s in str(args.haz_what).split(",") if s.strip()}

    # hazard tags to map
    haz_tags = []
    if "haz" in map_set:
        if args.haz_thr is not None:
            hs = [float(x) for x in str(args.haz_thr).split(",") if x.strip() != ""]
            haz_tags = [eta_tag_cm(h) for h in hs]
        else:
            # if user didn't specify, infer from columns
            haz_tags = infer_haz_tags_from_df(df)

        if not haz_tags:
            raise SystemExit("map-set includes 'haz' but no hazard thresholds found. "
                            "Either pass --haz-thr (e.g. 0.5) or ensure summary has AERH*/RPH* columns.")
        
    # RPs: either user-specified or inferred from summary
    if args.rps:
        rps = _sort_rps([s.strip() for s in str(args.rps).split(",") if s.strip() != ""])
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

    # -----------------------------
    # LABELING: per-map or global
    # -----------------------------
    label_mode = args.label_mode
    if label_mode is None:
        label_mode = "global" if args.label_by else "per-map"

    label_ids_g = label_x_g = label_y_g = None
    label_cache = {}

    if args.label_top_n <= 0:
        print("[INFO] labeling disabled (--label-top-n <= 0)")
        label_mode = "none"

    if label_mode == "global":
        if args.label_by is None:
            print("[WARN] label-mode=global but --label-by not set -> disabling labels")
            label_mode = "none"
        else:
            label_ids_g, label_x_g, label_y_g = pick_label_points(df, args.label_top_n, args.label_by)
            if label_ids_g is not None:
                print(f"[INFO] labeling (global) top {args.label_top_n} by: {args.label_by}")
            else:
                print(f"[WARN] labeling (global) failed for column: {args.label_by} -> disabling labels")
                label_mode = "none"

    elif label_mode == "per-map":
        print(f"[INFO] labeling (per-map): top {args.label_top_n} by each map variable")

    def get_labels(by_col):
        if label_mode == "none":
            return (None, None, None)
        if label_mode == "global":
            return (label_ids_g, label_x_g, label_y_g)
        # per-map
        if by_col in label_cache:
            return label_cache[by_col]
        labs = pick_label_points(df, args.label_top_n, by_col)
        label_cache[by_col] = labs
        return labs

    mapped_cols = []

    # -----------------------------
    # MAPS
    # -----------------------------
    for rp in rps:
        for typ in types:

            if "fail" in map_set:
                pf_col = f"Pf_{rp}_{typ}"
                if pf_col in df.columns:
                    lab_ids, lab_x, lab_y = get_labels(pf_col)
                    outpng = os.path.join(out_dir, f"map_{pf_col}.png")
                    plot_points_on_bathy(
                        Zb, extent, x, y, df[pf_col].to_numpy(float),
                        title=pf_col, outpng=outpng,
                        label_ids=lab_ids, label_x=lab_x, label_y=lab_y,
                        cbar_label=pf_col
                    )
                    mapped_cols.append(pf_col)

                for p in eta_pcts:
                    eta_col = f"eta0_fail{rp}_p{p:02d}_m_{typ}"
                    if eta_col in df.columns:
                        lab_ids, lab_x, lab_y = get_labels(eta_col)
                        outpng = os.path.join(out_dir, f"map_{eta_col}.png")
                        plot_points_on_bathy(
                            Zb, extent, x, y, df[eta_col].to_numpy(float),
                            title=eta_col, outpng=outpng,
                            label_ids=lab_ids, label_x=lab_x, label_y=lab_y,
                            cbar_label=eta_col
                        )
                        mapped_cols.append(eta_col)

            if "tsu" in map_set:
                pf_col = f"PfT_{rp}_{typ}"
                if pf_col in df.columns:
                    lab_ids, lab_x, lab_y = get_labels(pf_col)
                    outpng = os.path.join(out_dir, f"map_{pf_col}.png")
                    plot_points_on_bathy(
                        Zb, extent, x, y, df[pf_col].to_numpy(float),
                        title=pf_col, outpng=outpng,
                        label_ids=lab_ids, label_x=lab_x, label_y=lab_y,
                        cbar_label=pf_col
                    )
                    mapped_cols.append(pf_col)

                for p in eta_pcts:
                    eta_col = f"eta0_tsu{rp}_p{p:02d}_m_{typ}"
                    if eta_col in df.columns:
                        lab_ids, lab_x, lab_y = get_labels(eta_col)
                        outpng = os.path.join(out_dir, f"map_{eta_col}.png")
                        plot_points_on_bathy(
                            Zb, extent, x, y, df[eta_col].to_numpy(float),
                            title=eta_col, outpng=outpng,
                            label_ids=lab_ids, label_x=lab_x, label_y=lab_y,
                            cbar_label=eta_col
                        )
                        mapped_cols.append(eta_col)

            if "smf" in map_set:
                c1 = f"PSMF_{rp}_{typ}"
                if c1 in df.columns:
                    lab_ids, lab_x, lab_y = get_labels(c1)
                    outpng = os.path.join(out_dir, f"map_{c1}.png")
                    plot_points_on_bathy(
                        Zb, extent, x, y, df[c1].to_numpy(float),
                        title=c1, outpng=outpng,
                        label_ids=lab_ids, label_x=lab_x, label_y=lab_y,
                        cbar_label=c1
                    )
                    mapped_cols.append(c1)

                c2 = f"RP_SMF_{rp}_{typ}"
                if c2 in df.columns:
                    lab_ids, lab_x, lab_y = get_labels(c2)
                    outpng = os.path.join(out_dir, f"map_{c2}.png")
                    plot_points_on_bathy(
                        Zb, extent, x, y, df[c2].to_numpy(float),
                        title=c2, outpng=outpng,
                        label_ids=lab_ids, label_x=lab_x, label_y=lab_y,
                        cbar_label=c2
                    )
                    mapped_cols.append(c2)
                    
        # -----------------------------
    # HAZARD MAPS (
    # -----------------------------
    if "haz" in map_set:
        for tag in haz_tags:
            for typ in types:

                # label mode: per-map labels follow the mapped column
                def labels_for(colname):
                    if args.label_mode == "per-map":
                        asc = colname.startswith("RPH")  # for return period, "most hazardous" = smallest RP
                        return pick_label_points(df, args.label_top_n, colname, ascending=asc)
                    else:
                        return (label_ids, label_x, label_y)

                if "aer" in haz_what:
                    col = f"AERH{tag}_{typ}"
                    if col in df.columns:
                        lid, lx, ly = labels_for(col)
                        outpng = os.path.join(out_dir, f"map_{col}.png")
                        plot_points_on_bathy(
                            Zb, extent, x, y, df[col].to_numpy(float),
                            title=col, outpng=outpng,
                            label_ids=lid, label_x=lx, label_y=ly,
                            cbar_label=col
                        )
                        mapped_cols.append(col)

                if "rp" in haz_what:
                    col = f"RPH{tag}_{typ}"
                    if col in df.columns:
                        lid, lx, ly = labels_for(col)
                        outpng = os.path.join(out_dir, f"map_{col}.png")
                        plot_points_on_bathy(
                            Zb, extent, x, y, df[col].to_numpy(float),
                            title=col, outpng=outpng,
                            label_ids=lid, label_x=lx, label_y=ly,
                            cbar_label=col
                        )
                        mapped_cols.append(col)

                        
    inv = pd.DataFrame({"mapped_column": mapped_cols})
    inv_path = os.path.join(out_dir, "maps_inventory.csv")
    inv.to_csv(inv_path, index=False)
    print("\n[DONE] maps in:", out_dir)
    print("[DONE] inventory:", inv_path)


if __name__ == "__main__":
    main()