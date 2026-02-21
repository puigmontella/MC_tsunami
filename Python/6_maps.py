#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# project root on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.insert(0, ROOT)

from utilities.config import load_config, abspath


def read_bathy_for_plot(bathy_tif):
    if not bathy_tif or (not os.path.exists(bathy_tif)):
        return None
    with rasterio.open(bathy_tif) as src:
        Z = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None and not (isinstance(nod, float) and np.isnan(nod)):
            Z[Z == nod] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    return Z, extent


def point_map(Z_extent, x, y, c, title, out_png, cbar_label):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(10, 10))

    if Z_extent is not None:
        Z, extent = Z_extent
        plt.imshow(np.ma.masked_invalid(Z), extent=extent, origin="upper")
        plt.colorbar(label="bathy (m)")

    sc = plt.scatter(x, y, c=c, s=40, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, label=cbar_label)
    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()
    print("[OK] wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description="Map Pf/PfT/eta0 metrics from summary_eq_center_all_rp.csv")
    ap.add_argument("--config", required=True, help="cases/config.json")
    ap.add_argument("--rps", default="100,200,475,750", help="Comma list of RPs to map")
    ap.add_argument("--types", default="rot", help="Comma list among: rot, trans, any")
    ap.add_argument("--what", default="PfT", choices=["Pf", "PfT", "eta0_fail", "eta0_tsu"],
                    help="Which metric family to map")
    ap.add_argument("--eta-pct", type=int, default=99, help="eta0 percentile (50/90/95/99)")
    ap.add_argument("--out-dir", default=None, help="Override maps dir (else cfg outputs.maps_dir or transects_dir)")

    args = ap.parse_args()
    cfg = load_config(args.config)

    transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"])
    index_csv = os.path.join(transects_dir, "index.csv")

    # This is the file your step 5 wrote (or cfg outputs.summary_csv points to)
    summary_csv = abspath(cfg, cfg["outputs"].get("summary_csv", "cases/tenerife/outputs/summary_eq_center_all_rp.csv"))

    if not os.path.exists(index_csv):
        raise SystemExit("Missing index.csv: " + index_csv)
    if not os.path.exists(summary_csv):
        raise SystemExit("Missing summary CSV: " + summary_csv)

    maps_dir = args.out_dir
    if maps_dir is None:
        maps_dir = abspath(cfg, cfg["outputs"].get("maps_dir", transects_dir))
    os.makedirs(maps_dir, exist_ok=True)

    bathy_tif = abspath(cfg, cfg["paths"].get("bathy_tif", "")) if "paths" in cfg else ""
    Z_extent = read_bathy_for_plot(bathy_tif)

    idx = pd.read_csv(index_csv).rename(columns={"id": "transect_id"})
    summ = pd.read_csv(summary_csv)

    # join
    df = idx.merge(summ, on="transect_id", how="inner")
    if not {"x0", "y0"}.issubset(df.columns):
        raise SystemExit("index.csv must contain x0,y0. Found: " + ",".join(df.columns))

    rps = [int(x.strip()) for x in str(args.rps).split(",") if x.strip()]
    types = [x.strip() for x in str(args.types).split(",") if x.strip()]

    for typ in types:
        for rp in rps:
            if args.what in ("Pf", "PfT"):
                col = f"{args.what}_{rp}_{typ}"
                if col not in df.columns:
                    print("[WARN] missing column:", col)
                    continue
                c = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
                out_png = os.path.join(maps_dir, f"map_{args.what}_RP{rp}_{typ}.png")
                point_map(Z_extent, df["x0"], df["y0"], c,
                          f"{args.what} (RP={rp} yr) — {typ}",
                          out_png, col)

                # optional: annualized rate (useful for comparing RPs)
                lam = c / float(rp)
                out_png2 = os.path.join(maps_dir, f"map_lambda_{args.what}_RP{rp}_{typ}.png")
                point_map(Z_extent, df["x0"], df["y0"], lam,
                          f"lambda={args.what}/RP (RP={rp} yr) — {typ}",
                          out_png2, f"lambda_{col}")

            else:
                # eta0 columns in your file look like:
                # eta0_fail_RP100_p99_m_rot
                fam = args.what
                pct = int(args.eta_pct)
                col = f"{fam}_RP{rp}_p{pct}_m_{typ}"
                if col not in df.columns:
                    print("[WARN] missing column:", col)
                    continue
                c = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
                out_png = os.path.join(maps_dir, f"map_{fam}_RP{rp}_p{pct}_{typ}.png")
                point_map(Z_extent, df["x0"], df["y0"], c,
                          f"{fam} p{pct} (RP={rp} yr) — {typ}",
                          out_png, col)

    print("[DONE] maps in:", maps_dir)


if __name__ == "__main__":
    main()