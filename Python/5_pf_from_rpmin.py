#!/usr/bin/env python3
"""
5_pf_from_rpmin.py

Build Pf(RP) curves from RPmin_yr in MC outputs:
  Pf(RP)  = P(RPmin_yr <= RP) among ALL accepted trials
  PfT(RP) = P(RPmin_yr <= RP AND eta0_m >= eta_min) (optional)

Supports:
  --level transects   -> reads <transects_dir>/tr_*_{rot,trans}_mcs.csv
  --level areas       -> reads <areas_dir>/ar_*_{rot,trans}_mcs.csv

Outputs:
  <level_dir>/summary_pf.csv
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd

from utilities.stats import parse_transect_id, quantile_safe


def parse_area_id(path: str) -> int:
    b = os.path.basename(path)
    # matches ar_0059_... or ar_000059_...
    import re
    m = re.search(r"ar_(\d+)", b)
    if not m:
        raise ValueError(f"Cannot parse area id from: {path}")
    return int(m.group(1))


def rp_label(RP: float) -> str:
    """Stable label for columns."""
    if np.isfinite(RP) and abs(RP - round(RP)) < 1e-6:
        return str(int(round(RP)))
    return f"{RP:g}"


def compute_pf_curves(df, rps_out, eta_min, pcts):
    out = {}

    rpmin = pd.to_numeric(df.get("RPmin_yr", np.nan), errors="coerce").to_numpy(dtype=float)
    eta0  = pd.to_numeric(df.get("eta0_m", np.nan), errors="coerce").to_numpy(dtype=float)

    out["n_trials"] = int(len(df))

    if "failed_within_cap" in df.columns:
        fcap = pd.to_numeric(df["failed_within_cap"], errors="coerce").fillna(0).to_numpy(dtype=int)
        m_valid = (fcap == 1) & np.isfinite(rpmin)
    else:
        m_valid = np.isfinite(rpmin)

    out["n_failed_within_cap"] = int(np.sum(m_valid))
    out["fail_frac_within_cap"] = float(np.mean(m_valid)) if len(df) else np.nan

    for RP in rps_out:
        RP = float(RP)
        lab = rp_label(RP)

        fail_by_RP = m_valid & (rpmin <= RP)
        out[f"Pf_{lab}"] = float(np.mean(fail_by_RP)) if len(df) else np.nan

        ef = eta0[fail_by_RP]
        for p in pcts:
            out[f"eta0_fail_RP{lab}_p{int(100*p):02d}_m"] = quantile_safe(ef, p)

        if eta_min is not None and np.isfinite(eta_min):
            tsu = fail_by_RP & np.isfinite(eta0) & (eta0 >= float(eta_min))
            out[f"PfT_{lab}"] = float(np.mean(tsu)) if len(df) else np.nan

            et = eta0[tsu]
            for p in pcts:
                out[f"eta0_tsu_RP{lab}_p{int(100*p):02d}_m"] = quantile_safe(et, p)

    return out


def main():
    ap = argparse.ArgumentParser(description="Build Pf(RP) curves from RPmin_yr in MC outputs.")
    ap.add_argument("--config", default="config.json", help="Path to case config.json")
    ap.add_argument("--level", default="transects", choices=["transects", "areas"],
                    help="Summarize transects or areas.")
    ap.add_argument("--pcts", default="50,90,95,99", help="Comma list of percentiles for eta0 stats.")
    ap.add_argument("--types", default="rot,trans,any",
                    help="Which types to include: rot,trans,any (comma list). Default: rot,trans,any")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # pick directory + filename patterns
    if args.level == "transects":
        base_dir = cfg["outputs"]["transects_dir"]
        pat_rot = "tr_*_rot_mcs.csv"
        pat_trans = "tr_*_trans_mcs.csv"
        parse_id = parse_transect_id
        id_col = "transect_id"
    else:
        base_dir = cfg["outputs"]["areas_dir"]
        pat_rot = "ar_*_rot_mcs.csv"
        pat_trans = "ar_*_trans_mcs.csv"
        parse_id = parse_area_id
        id_col = "area_id"

    out_csv = os.path.join(base_dir, "summary_pf.csv")

    # RPs to report (keys of pga_rasters)
    pga_dict = cfg["paths"]["pga_rasters"]
    rps_out = sorted([float(k) for k in pga_dict.keys()])

    # Tsunami threshold (optional)
    eta_min = None
    try:
        eta_min = cfg.get("pga_loading", {}).get("eta_min_m", None)
        if eta_min is not None:
            eta_min = float(eta_min)
    except Exception:
        eta_min = None

    pcts = [float(x) / 100.0 for x in str(args.pcts).split(",") if x.strip() != ""]
    types = [t.strip() for t in args.types.split(",") if t.strip() != ""]

    rot_files = sorted(glob.glob(os.path.join(base_dir, pat_rot)))
    trans_files = sorted(glob.glob(os.path.join(base_dir, pat_trans)))

    rot_map = {parse_id(f): f for f in rot_files}
    trans_map = {parse_id(f): f for f in trans_files}

    ids = sorted(set(rot_map.keys()) | set(trans_map.keys()))
    if not ids:
        raise SystemExit(f"[FAIL] No MC files found in: {base_dir}")

    rows = []
    for _id in ids:
        row = {id_col: int(_id)}

        df_rot = None
        df_tr = None

        if ("rot" in types or "any" in types) and _id in rot_map:
            df_rot = pd.read_csv(rot_map[_id])
        elif ("rot" in types or "any" in types):
            print(f"[WARN] missing rot file for {id_col}={_id} (skipping rot/any contribution)")

        if ("trans" in types or "any" in types) and _id in trans_map:
            df_tr = pd.read_csv(trans_map[_id])
        elif ("trans" in types or "any" in types):
            print(f"[WARN] missing trans file for {id_col}={_id} (skipping trans/any contribution)")

        if "rot" in types and df_rot is not None and len(df_rot):
            d = compute_pf_curves(df_rot, rps_out, eta_min, pcts)
            for k, v in d.items():
                row[f"{k}_rot"] = v

        if "trans" in types and df_tr is not None and len(df_tr):
            d = compute_pf_curves(df_tr, rps_out, eta_min, pcts)
            for k, v in d.items():
                row[f"{k}_trans"] = v

        # pooled "any" (concatenate available parts)
        if "any" in types:
            parts = [x for x in [df_rot, df_tr] if x is not None and len(x)]
            if parts:
                df_any = pd.concat(parts, ignore_index=True)
                d = compute_pf_curves(df_any, rps_out, eta_min, pcts)
                for k, v in d.items():
                    row[f"{k}_any"] = v

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(id_col).reset_index(drop=True)

    # cast counts to Int64
    for c in out.columns:
        if c == id_col or c.startswith("n_trials") or c.startswith("n_failed_within_cap"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    out.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"[OK] wrote {out_csv}")
    print("[INFO] level:", args.level)
    print("[INFO] RPs reported:", ", ".join(rp_label(rp) for rp in rps_out))
    if eta_min is not None and np.isfinite(eta_min):
        print("[INFO] eta_min_m:", eta_min)


if __name__ == "__main__":
    main()