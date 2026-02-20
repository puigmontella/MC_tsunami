#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utilities.config import load_config, abspath
from utilities.stats import find_col, quantile_safe


def _summ(x, name):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    print(f"\n=== {name} ===")
    if x.size == 0:
        print("  (empty)")
        return
    stats = {
        "n": x.size,
        "min": float(np.min(x)),
        "p05": quantile_safe(x, 0.05),
        "p50": quantile_safe(x, 0.50),
        "p90": quantile_safe(x, 0.90),
        "p95": quantile_safe(x, 0.95),
        "p99": quantile_safe(x, 0.99),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }
    for k, v in stats.items():
        if np.isfinite(v):
            print(f"  {k:>5}: {v:.6g}")
        else:
            print(f"  {k:>5}: nan")


def process_one(inp, l_max, t_max, end_margin, overwrite=False, verbose=True):
    df = pd.read_csv(inp)
    if df.empty:
        if verbose:
            print("[SKIP] empty:", inp)
        return None

    col_s1 = find_col(df, ["s1_m"])
    col_k  = find_col(df, ["kcrit"])
    col_eta = find_col(df, ["eta0_m"])

    col_L = find_col(df, ["L_eff_m", "l_eff_m", "L_m", "l_m"])
    col_T = find_col(df, ["T_m", "T_max_m", "Tmax_m", "T"])

    if col_s1 is None or col_L is None or col_T is None:
        raise SystemExit(
            f"{os.path.basename(inp)} missing required columns. "
            f"Need s1 + (L or L_eff) + (T or T_max). Has: {list(df.columns)}"
        )

    s1 = df[col_s1].to_numpy(float)
    L  = df[col_L].to_numpy(float)
    T  = df[col_T].to_numpy(float)

    s_end = float(np.nanmax(s1[np.isfinite(s1)])) if np.isfinite(s1).any() else np.nan
    if not np.isfinite(s_end):
        if verbose:
            print("[SKIP] no finite s1_m:", inp)
        return None

    m = np.isfinite(s1) & np.isfinite(L) & np.isfinite(T)
    m &= (L <= float(l_max))
    m &= (T <= float(t_max))
    m &= (s1 <= (s_end - float(end_margin)))

    df_reg = df[m].copy()

    out = inp.replace(".csv", "_REGIONAL.csv")
    if (not overwrite) and os.path.exists(out):
        if verbose:
            print("[INFO] exists (skip write):", out)
        return out

    df_reg.to_csv(out, index=False)

    if verbose:
        print(f"\n[OK] regional subset: {len(df_reg)} / {len(df)}  -> {out}")
        if col_k is not None:
            _summ(df[col_k].to_numpy(float), "kcrit (ALL)")
            _summ(df_reg[col_k].to_numpy(float), "kcrit (REGIONAL)")
        if col_eta is not None:
            _summ(df[col_eta].to_numpy(float), "eta0_m (ALL)")
            _summ(df_reg[col_eta].to_numpy(float), "eta0_m (REGIONAL)")

    return out


def main():
    ap = argparse.ArgumentParser(description="Step 4: Regional filter for MC outputs (config-driven).")

    ap.add_argument("--config", required=True, help="Path to case config.json")

    # New flag
    ap.add_argument("--type", default="both", choices=["trans", "rot", "both"],
                    help="Which MC outputs to filter.")
    # Backwards-compatible alias (so run_pipeline.py using --mode keeps working)
    ap.add_argument("--mode", dest="type", choices=["trans", "rot", "both"],
                    help="Alias for --type (kept for older pipelines).")

    ap.add_argument("inp", nargs="?", default=None,
                    help="Optional single input CSV. If omitted, uses patterns in transects dir.")
    ap.add_argument("--pattern", default=None,
                    help="Optional glob pattern override (e.g. '.../tr_*_grilli_mcs.csv').")

    ap.add_argument("--l-max", type=float, default=None, help="Max slide length (m)")
    ap.add_argument("--t-max", type=float, default=None, help="Max thickness (m)")
    ap.add_argument("--end-margin", type=float, default=None, help="Exclude last <margin> meters of transect end (m)")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite *_REGIONAL.csv if exists")
    ap.add_argument("--quiet", action="store_true", help="Less printing")

    args = ap.parse_args()

    cfg = load_config(args.config)

    transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"])

    reg = cfg.get("regional_filter", {})
    l_max = float(args.l_max) if args.l_max is not None else float(reg.get("L_max", 8000.0))
    t_max = float(args.t_max) if args.t_max is not None else float(reg.get("T_max", 150.0))
    end_margin = float(args.end_margin) if args.end_margin is not None else float(reg.get("end_margin", 200.0))

    verbose = (not args.quiet)

    # Decide which files to filter
    files = []
    if args.pattern:
        files = sorted(glob.glob(args.pattern))
        if not files:
            raise SystemExit("No files for pattern: {}".format(args.pattern))
    elif args.inp:
        files = [args.inp]
    else:
        pats = []
        if args.type in ("trans", "both"):
            pats.append(os.path.join(transects_dir, "tr_*_grilli_mcs.csv"))
        if args.type in ("rot", "both"):
            pats.append(os.path.join(transects_dir, "tr_*_grilli_rot_mcs.csv"))

        for p in pats:
            files.extend(sorted(glob.glob(p)))

        if not files:
            raise SystemExit(
                f"No MC files found in {transects_dir} for type='{args.type}'. "
                "Expected e.g. tr_*_grilli_mcs.csv / tr_*_grilli_rot_mcs.csv"
            )

    if verbose:
        print("[INFO] transects_dir:", transects_dir)
        print("[INFO] filtering type:", args.type)
        print("[INFO] regional filter params: L_max={}, T_max={}, end_margin={}".format(l_max, t_max, end_margin))
        print("[INFO] files:", len(files))

    for f in files:
        process_one(
            f,
            l_max=l_max,
            t_max=t_max,
            end_margin=end_margin,
            overwrite=args.overwrite,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
