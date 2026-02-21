#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

# Project root on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utilities.config import load_config, abspath
from utilities.stats import parse_transect_id


def rp_sort_key(x):
    try:
        return float(x)
    except Exception:
        return 1e99


def safe_quantile(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.quantile(a, q))


def load_mc_csv(path):
    """Robust read: skip empty/headers-only files."""
    if not path or (not os.path.exists(path)):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df is None or df.shape[1] == 0 or len(df) == 0:
        return None
    return df


def get_numeric_array(df: pd.DataFrame, col: str, default=np.nan):
    """Return a float array of length len(df). If col missing, returns all default."""
    if (df is None) or (len(df) == 0):
        return np.asarray([], float)
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def compute_pf_curves(df, rps_out, eta_min=None, pcts=(0.5, 0.9, 0.95, 0.99)):
    """
    Pf(RP) = P(RPmin <= RP) using RPmin_yr column.
    Uses failed_within_cap if present (else assumes RPmin valid whenever finite).
    Optionally PfT(RP) = P(RPmin <= RP AND eta0 >= eta_min).
    Also eta0 quantiles among failures-by-RP set.
    """
    out = {}

    rpmin = get_numeric_array(df, "RPmin_yr", default=np.nan)
    eta0  = get_numeric_array(df, "eta0_m", default=np.nan)

    n = int(len(df))
    out["n_trials"] = n
    out["n_rpmin_finite"] = int(np.sum(np.isfinite(rpmin)))

    # valid failures within cap (if that flag exists)
    if "failed_within_cap" in df.columns:
        fcap = pd.to_numeric(df["failed_within_cap"], errors="coerce").fillna(0).to_numpy(dtype=int)
        m_valid = (fcap == 1) & np.isfinite(rpmin)
        out["n_failed_within_cap"] = int(np.sum(fcap == 1))
    else:
        m_valid = np.isfinite(rpmin)
        out["n_failed_within_cap"] = int(np.sum(m_valid))

    # Quick FS0 diagnostic if present
    if "FS0" in df.columns:
        FS0 = get_numeric_array(df, "FS0", default=np.nan)
        out["FS0_median"] = float(np.nanmedian(FS0)) if np.isfinite(FS0).any() else np.nan

    # If RPmin is missing everywhere, emit NaNs for Pf columns (instead of crashing)
    if not np.isfinite(rpmin).any():
        for RP in rps_out:
            RP = float(RP)
            out[f"Pf_{int(RP)}"] = np.nan
            for p in pcts:
                out[f"eta0_fail_RP{int(RP)}_p{int(100*p):02d}_m"] = np.nan
            if eta_min is not None and np.isfinite(eta_min):
                out[f"PfT_{int(RP)}"] = np.nan
                for p in pcts:
                    out[f"eta0_tsu_RP{int(RP)}_p{int(100*p):02d}_m"] = np.nan
        return out

    for RP in rps_out:
        RP = float(RP)

        fail_by_RP = m_valid & (rpmin <= RP)
        Pf = float(np.mean(fail_by_RP))  # denominator = all accepted trials
        out[f"Pf_{int(RP)}"] = Pf

        # eta0 stats among failures-by-RP
        ef = eta0[fail_by_RP]
        for p in pcts:
            out[f"eta0_fail_RP{int(RP)}_p{int(100*p):02d}_m"] = safe_quantile(ef, p)

        if eta_min is not None and np.isfinite(eta_min):
            tsu = fail_by_RP & np.isfinite(eta0) & (eta0 >= float(eta_min))
            out[f"PfT_{int(RP)}"] = float(np.mean(tsu))

            et = eta0[tsu]
            for p in pcts:
                out[f"eta0_tsu_RP{int(RP)}_p{int(100*p):02d}_m"] = safe_quantile(et, p)

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Step 5 (Grilli-style): build Pf(RP) curves from RPmin_yr in MC outputs (no raster resampling)."
    )
    ap.add_argument("--config", required=True, help="cases/config.json")
    ap.add_argument("--rps", default=None,
                    help="Comma list of RPs (years) to report Pf at.")
    ap.add_argument("--eta-min", type=float, default=None,
                    help="Optional tsunamigenic threshold (m): also compute PfT_* and eta0_tsu_* stats.")
    ap.add_argument("--pcts", default="50,90,95,99", help="Comma list of percentiles for eta0 stats.")
    ap.add_argument("--out-csv", default=None, help="Override output CSV path.")
    ap.add_argument("--types", default="rot,trans,any",
                    help="Which types to include: rot,trans,any (comma list). Default: rot,trans,any")
    ap.add_argument("--diagnose-missing", action="store_true",
                    help="Print which files are missing RPmin_yr and continue (those get NaNs).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"])

    # Output file
    if args.out_csv:
        out_csv = os.path.abspath(args.out_csv)
    else:
        out_csv = abspath(cfg, cfg["outputs"].get("summary_csv", os.path.join(transects_dir, "summary_pf.csv")))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # RP list
    if args.rps:
        rps_out = [float(x) for x in args.rps.split(",") if x.strip() != ""]
    else:
        pga_rasters = cfg.get("paths", {}).get("pga_rasters", {})
        rps_out = [float(k) for k in pga_rasters.keys()] if isinstance(pga_rasters, dict) else [100.0, 200.0, 475.0, 750.0]
    rps_out = sorted(set(rps_out), key=rp_sort_key)

    pcts = [float(x)/100.0 for x in str(args.pcts).split(",") if x.strip() != ""]
    types = [t.strip() for t in args.types.split(",") if t.strip() != ""]

    # Discover MC files
    rot_files   = sorted(glob.glob(os.path.join(transects_dir, "tr_*_grilli_rot_mcs.csv")))
    trans_files = sorted(glob.glob(os.path.join(transects_dir, "tr_*_grilli_mcs.csv")))

    rot_map   = {parse_transect_id(f): f for f in rot_files}
    trans_map = {parse_transect_id(f): f for f in trans_files}

    tids = sorted(set(list(rot_map.keys()) + list(trans_map.keys())))
    if not tids:
        raise SystemExit(f"No MC files found in: {transects_dir}")

    rows = []
    missing = []

    for tid in tids:
        row = {"transect_id": int(tid)}

        df_rot = load_mc_csv(rot_map.get(tid, "")) if ("rot" in types or "any" in types) else None
        df_tr  = load_mc_csv(trans_map.get(tid, "")) if ("trans" in types or "any" in types) else None

        # rot
        if "rot" in types and df_rot is not None:
            if ("RPmin_yr" not in df_rot.columns) and args.diagnose_missing:
                missing.append(rot_map.get(tid, ""))
            d = compute_pf_curves(df_rot, rps_out, eta_min=args.eta_min, pcts=pcts)
            for k, v in d.items():
                row[f"{k}_rot"] = v

        # trans
        if "trans" in types and df_tr is not None:
            if ("RPmin_yr" not in df_tr.columns) and args.diagnose_missing:
                missing.append(trans_map.get(tid, ""))
            d = compute_pf_curves(df_tr, rps_out, eta_min=args.eta_min, pcts=pcts)
            for k, v in d.items():
                row[f"{k}_trans"] = v

        # any
        if "any" in types and (df_rot is not None or df_tr is not None):
            parts = []
            if df_rot is not None:
                parts.append(df_rot)
            if df_tr is not None:
                parts.append(df_tr)
            df_any = pd.concat(parts, ignore_index=True) if parts else None
            if df_any is not None and len(df_any) > 0:
                d = compute_pf_curves(df_any, rps_out, eta_min=args.eta_min, pcts=pcts)
                for k, v in d.items():
                    row[f"{k}_any"] = v

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("transect_id").reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")
    print("[INFO] RPs reported:", ", ".join(str(int(rp)) for rp in rps_out))

    if args.diagnose_missing and missing:
        print("\n[WARN] files missing RPmin_yr (these get NaNs):")
        for f in sorted(set(missing)):
            print(" ", f)


if __name__ == "__main__":
    main()