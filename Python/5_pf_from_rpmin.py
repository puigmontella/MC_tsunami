#!/usr/bin/env python3
"""
5_summary_pf.py

Build Pf(RP) curves from RPmin_yr in MC outputs:
  Pf(RP)  = P(RPmin_yr <= RP) among accepted trials
  PfT(RP) = P(RPmin_yr <= RP AND eta0_m >= eta_min) (optional)

Also reports eta0 quantiles among failures-by-RP (and among tsunami-eligible if eta_min set).

Outputs:
  <transects_dir>/summary_pf.csv
"""



#!/usr/bin/env python3
import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd


from utilities.stats import parse_transect_id, quantile_safe


### functions


def compute_pf_curves(df, rps_out, eta_min, pcts):
    """
    Given ONE transect's MC output dataframe, compute probability-of-failure curves vs RP.

    Inputs
    ------
    df      : MC output for one transect (rot OR trans OR merged "any")
              Must contain RPmin_yr and eta0_m.
    rps_out : list of return periods (years) at which to report Pf (e.g. [100, 200, 475, 2475])
    eta_min : tsunami threshold (m). If not None, compute PfT and tsunami-conditioned eta0 quantiles.
    pcts    : list of quantiles in [0,1], e.g. [0.50, 0.90, 0.95, 0.99]

    Outputs
    -------
    out : dict of scalar metrics to be appended to the summary row.
    """
    out = {}

    # Convert columns  to numeric arrays
    rpmin = pd.to_numeric(df["RPmin_yr"], errors="coerce").to_numpy(dtype=float)
    eta0  = pd.to_numeric(df["eta0_m"], errors="coerce").to_numpy(dtype=float)

    # number of  accepted MC trials
    out["n_trials"] = int(len(df))

    # Decide which trials count as "valid failures within the RP cap"
    # If "failed_within_cap" exists, we restrict to those (and finite rpmin).
    # Otherwise, we assume any finite RPmin_yr indicates a failure within the ramp.
    if "failed_within_cap" in df.columns:
        fcap = pd.to_numeric(df["failed_within_cap"], errors="coerce").fillna(0).to_numpy(dtype=int)
        m_valid = (fcap == 1) & np.isfinite(rpmin)
        out["n_failed_within_cap"] = int(np.sum(fcap == 1))
    else:
        m_valid = np.isfinite(rpmin)
        out["n_failed_within_cap"] = int(np.sum(m_valid))


    # For each requested RP, compute:
    #   Pf(RP)  = fraction of ALL trials that have failed by that RP
    #          = mean( m_valid & (rpmin <= RP) )
    # Plus eta0 quantiles among the subset of trials that failed by RP.
    for RP in rps_out:
        RP = float(RP)

        fail_by_RP = m_valid & (rpmin <= RP)
        Pf = float(np.mean(fail_by_RP)) # P(RPmin <= RP) among valid failures
        out[f"Pf_{int(RP)}"] = Pf

        # eta0 stats among failures by RP
        ef = eta0[fail_by_RP]

        for p in pcts:
            out[f"eta0_fail_RP{int(RP)}_p{int(100*p):02d}_m"] = quantile_safe(ef, p)


        # optionl: PfT(RP) = P(RPmin <= RP AND eta0 >= eta_min
        #  PfT(RP) = fraction of ALL trials that failed by RP AND have eta0 above threshold

        if eta_min is not None and np.isfinite(eta_min):
            tsu = fail_by_RP & np.isfinite(eta0) & (eta0 >= float(eta_min))
            out[f"PfT_{int(RP)}"] = float(np.mean(tsu))

            et = eta0[tsu]
            for p in pcts:
                out[f"eta0_tsu_RP{int(RP)}_p{int(100*p):02d}_m"] = quantile_safe(et, p)

    return out


### main script ###

def main():
    ap = argparse.ArgumentParser(
        description="Step 5: build Pf(RP) curves from RPmin_yr in MC outputs"
    )
    ap.add_argument("--config", default="config.json", help="Path to case config.json")
    ap.add_argument("--pcts", default="50,90,95,99", help="Comma list of percentiles for eta0 stats.")
    ap.add_argument("--types", default="rot,trans,any",
                    help="Which types to include: rot,trans,any (comma list). Default: rot,trans,any")

    args = ap.parse_args()

    # --- Load case config  ---
    # Read case config
    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Directory where transect CSVs  
    transects_dir = cfg["outputs"]["transects_dir"]

    # Output file for the summary table 
    out_csv=transects_dir+"/summary_pf.csv"

    # We use the  pga_rasters as "the RP list to report"
    pga_dict = cfg["paths"]["pga_rasters"]  
    rps_out = sorted(pga_dict.keys(), key=float)

    # Tsunami threshold for eta0
    eta_min = cfg["pga_loading"]["eta_min_m"]  

    

    pcts = [float(x)/100.0 for x in str(args.pcts).split(",") if x.strip() != ""]
    types = [t.strip() for t in args.types.split(",") if t.strip() != ""]

    # find MC files
    rot_files   = sorted(glob.glob(os.path.join(transects_dir, "tr_*_rot_mcs.csv")))
    trans_files = sorted(glob.glob(os.path.join(transects_dir, "tr_*_trans_mcs.csv")))

    rot_map   = {parse_transect_id(f): f for f in rot_files}
    trans_map = {parse_transect_id(f): f for f in trans_files}

    tids = sorted(set(list(rot_map.keys()) + list(trans_map.keys())))
    if not tids:
        raise SystemExit(f"No MC files found in: {transects_dir}")
    
    # each element becomes one output row
    rows = []

    # Loop over transects and compute summary metrics
    for tid in tids:
        row = {"transect_id": int(tid)}

        path_rot= rot_map.get(tid, "")
        if not os.path.exists(path_rot) and ("rot" in types or "any" in types):
            raise SystemExit(f"[WARN] Missing rotational failure file for transect {tid}.")

        path_trans= trans_map.get(tid, "")
        if not os.path.exists(path_trans) and ("trans" in types or "any" in types):
            raise SystemExit(f"[WARN] Missing translational failure file for transect {tid}.")


        df_rot = pd.read_csv(rot_map.get(tid, "")) if ("rot" in types or "any" in types) else None
        df_tr  = pd.read_csv(trans_map.get(tid, "")) if ("trans" in types or "any" in types) else None

        # --- rotational summary ---
        if "rot" in types:
            d = compute_pf_curves(df_rot, rps_out, eta_min, pcts)
            for k, v in d.items():
                row[f"{k}_rot"] = v

        # --- translational summary ---
        if "trans" in types:
            if ("RPmin_yr" not in df_tr.columns):
                missing.append(trans_map.get(tid, ""))
            d = compute_pf_curves(df_tr, rps_out, eta_min, pcts)
            for k, v in d.items():
                row[f"{k}_trans"] = v

        # --- combined "any" = rot + trans together ---
        # This treats the union as one big MC sample for the transect.
        if "any" in types:
            parts = []
            if df_rot is not None:
                parts.append(df_rot)
            if df_tr is not None:
                parts.append(df_tr)
            df_any = pd.concat(parts, ignore_index=True) if parts else None
            if df_any is not None and len(df_any) > 0:
                d = compute_pf_curves(df_any, rps_out, eta_min, pcts)
                for k, v in d.items():
                    row[f"{k}_any"] = v

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("transect_id").reset_index(drop=True)
    for c in out.columns:
        if c == "transect_id" or c.startswith("n_trials") or c.startswith("n_failed_within_cap"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    out.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"[OK] wrote {out_csv}")
    print("[INFO] RPs reported:", ", ".join(str(int(rp)) for rp in rps_out))



if __name__ == "__main__":
    main()