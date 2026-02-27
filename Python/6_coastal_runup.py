#!/usr/bin/env python3
"""
Coastal runup proxy  from Monte Carlo source trials.

Reads:
  - <transects_dir>/index.csv                      (coastal points + normals)
  - <transects_dir>/tr_XXXX_trans_mcs.csv         (trans)
  - <transects_dir>/tr_XXXX_rot_mcs.csv     (rot)

Logic (for a chosen earthquake return period RP_eq):
  - Each accepted trial in a given transect file carries annual rate:
        w_event = (1/RP_eq) / N_accepted
    (and in "any" mode, multiply by mixture weights w_trans / w_rot)

  - Select only trials that:
        RPmin_yr <= RP_eq
        eta0_m   >= eta_min_src   (eta_min_src comes from config: pga_loading.eta_min_m)

  - Spread runup to all coastal receivers using angular Gaussian:
        W = exp(-0.5*(dtheta/sigma)^2)
    with hard cutoff |dtheta| > max_deg -> W=0
    and optional jitter on main direction (uniform in [-jitter, +jitter]).

  - Compute "design runup" for tsunami return periods T:
        find R_T such that sum_{events with R >= R_T} w_event >= 1/T

Outputs:
  - <coast_dir>/coast_runup_eqRP{RP}_{type}.csv
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd

from utilities.stats import weighted_percentile


# -----  functions -------

def design_runup_from_weighted_events(R, w, T_years):
    """
    Return design runup R_T such that weighted annual exceedance rate >= 1/T.
    Here w is an annual rate per event.
    """
    R = np.asarray(R, float)
    w = np.asarray(w, float)

    m = np.isfinite(R) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return 0.0

    Rm = R[m]
    wm = w[m]

    # Sort by decreasing runup so cumulative weights represent exceedance rate
    idx = np.argsort(Rm)[::-1]
    R_sorted = Rm[idx]
    w_sorted = wm[idx]

    cum = np.cumsum(w_sorted)
    target = 1.0 / float(T_years)

    j = int(np.searchsorted(cum, target, side="left"))
    if j >= R_sorted.size:
        return 0.0
    return round(float(R_sorted[j]),3)


def infer_source_xy_from_index(idx_row, s_center):
    """
    Use x0,y0,nx,ny + s_center to estimate source (slide center) x/y.
    Assumes transect points extend along the outward normal direction.
    """
    x0 = float(idx_row["x0"])
    y0 = float(idx_row["y0"])
    nx = float(idx_row["nx"])
    ny = float(idx_row["ny"])
    return (x0 + nx * float(s_center), y0 + ny * float(s_center))


  


######## main #############  
def main():
    ap = argparse.ArgumentParser(description="Coastal runup + design runup from MC trials.")
    ap.add_argument("--config", default="config.json", help="Path to case config.json")

    ap.add_argument("--types", default="any",
                    help="Comma list among: trans, rot, any (any combines trans+rot).")

    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # --- Load case config  ---
    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = json.load(f)


    # Directory where transect CSVs  
    transects_dir = cfg["outputs"]["transects_dir"]

    # Tsunami threshold for eta0
    eta_min_src = cfg["pga_loading"]["eta_min_m"]  

    # output dir
    coast_dir = cfg["outputs"]["coast_dir"]
    os.makedirs(coast_dir, exist_ok=True)



    runup_params_path = cfg["paths"]["parameters"]["coast_runup"]
    if not os.path.exists(runup_params_path):
        raise SystemExit(f"[FAIL] Missing coast_runup params JSON: {runup_params_path}")

    with open(runup_params_path, "r") as f:
        runup_params = json.load(f)

    eq_rp_yr = runup_params["eq_rp_yr"]
    tsu_rps = [int(x) for x in runup_params["tsu_rps_yr"]]
    types = [str(t).strip().lower() for t in runup_params["types"]]
    coast_eta_min = runup_params["coast_eta_min_m"]

    # angular spreading controls (degrees -> radians)
    sigma_deg = runup_params["sigma_deg"]
    max_deg = runup_params["max_deg"]
    jitter_deg = runup_params["jitter_deg"]
    sigma = np.radians(sigma_deg)
    max_a = np.radians(max_deg)
    jitter = np.radians(jitter_deg)

    # mixture weights (only used when type="any")
    w_rot = runup_params['w_rot']
    w_trans = runup_params['w_trans']   

    # RNG only used for jitter
    rng = np.random.default_rng()


    # -----------------------
    # Load coastal index points
    # -----------------------
    index_csv = os.path.join(transects_dir, "index.csv")
    idx = pd.read_csv(index_csv).rename(columns={"id": "transect_id"})

    coast_ids = idx["transect_id"].to_numpy(int)
    x_coast = idx["x0"].to_numpy(float)
    y_coast = idx["y0"].to_numpy(float)
    coast_s = idx["coast_s_m"].to_numpy(float)

    n_coast = x_coast.size
    if n_coast < 2:
        raise SystemExit("Need at least 2 coastal points in index.csv for alongshore spreading.")

    # -----------------------
    # Main loop over requested output "types"
    # -----------------------
    for typ_l in types:
        if typ_l not in ("trans", "rot", "any"):
            print("[WARN] unknown type:", typ_l, "skipping.")
            continue

        # Store runup samples per coastal receiver as lists (later concatenated)
        R_store = [[] for _ in range(n_coast)]
        W_store = [[] for _ in range(n_coast)]

        n_files_used = 0
        n_events_total = 0

        # Iterate each transect (= each coastal point + its normal)
        for _, row in idx.iterrows():
            tid = int(row["transect_id"])

            include_trans = (typ_l in ("trans", "any"))
            include_rot = (typ_l in ("rot", "any"))



            # helper: read + filter + spread one mechanism file
            def process_file(mcs_path, mix_weight):
                nonlocal n_files_used, n_events_total

                if not os.path.exists(mcs_path):
                    if args.verbose:
                        print("[WARN] missing:", mcs_path)
                    return

                df = pd.read_csv(mcs_path)
                if df is None or df.empty:
                    return

                n_acc = int(len(df))
                if n_acc <= 0:
                    return

                # Check columns that are needed to compute filtering + geometry
                needed = ("RPmin_yr", "eta0_m", "s0_m", "s1_m")
                for col in needed:
                    if col not in df.columns:
                        print("[WARN] missing", col, "in", os.path.basename(mcs_path))
                        return

                RPmin = pd.to_numeric(df["RPmin_yr"], errors="coerce").to_numpy(float)
                eta0 = pd.to_numeric(df["eta0_m"], errors="coerce").to_numpy(float)
                s0 = pd.to_numeric(df["s0_m"], errors="coerce").to_numpy(float)
                s1 = pd.to_numeric(df["s1_m"], errors="coerce").to_numpy(float)

                # Select events that fail by eq RP AND are tsunamigenic at source
                m = (
                    np.isfinite(RPmin) & (RPmin <= eq_rp_yr) &
                    np.isfinite(eta0) & (eta0 >= eta_min_src)
                )
                if not np.any(m):
                    return

                # Annual rate weight per accepted trial in this file
                # (mixture factor applied here if typ="any")
                w_event = mix_weight * (1.0 / eq_rp_yr) / float(n_acc)

                # Subset arrays to selected events
                eta0_sel = eta0[m]
                s_center = 0.5 * (s0[m] + s1[m])

                # Infer source locations from transect normal info
                x_src = np.empty_like(s_center, dtype=float)
                y_src = np.empty_like(s_center, dtype=float)
                for i in range(s_center.size):
                    xs, ys = infer_source_xy_from_index(row, s_center[i])
                    x_src[i] = xs
                    y_src[i] = ys

                # Main direction: source -> its own coastal anchor point (x0,y0)
                x0 = float(row["x0"])
                y0 = float(row["y0"])
                theta0 = np.arctan2((y0 - y_src), (x0 - x_src))

                # Optional jitter on the main direction
                if jitter > 0:
                    theta0 = theta0 + rng.uniform(-jitter, +jitter, size=theta0.shape)

                # Broadcast to all coastal receivers 
                dx = x_coast[None, :] - x_src[:, None]
                dy = y_coast[None, :] - y_src[:, None]
                theta = np.arctan2(dy, dx)

                # Angle difference wrapped to [-pi, pi] 
                dtheta = np.arctan2(np.sin(theta - theta0[:, None]), np.cos(theta - theta0[:, None]))

                # Angular weights (Gaussian + hard cutoff)
                if sigma > 0:
                    W = np.exp(-0.5 * (dtheta / sigma) ** 2)
                else:
                    W = (np.abs(dtheta) <= max_a).astype(float)

                if max_a > 0:
                    W[np.abs(dtheta) > max_a] = 0.0

                # Runup proxy at coast = eta0 * angular weight
                R = (eta0_sel[:, None] * W).astype(np.float32)

                # Each selected event has same weight for all receivers
                ww = np.full((eta0_sel.size,), w_event, dtype=np.float32)

                # Append per-coast lists
                for j in range(n_coast):
                    R_store[j].append(R[:, j])
                    W_store[j].append(ww)

                n_files_used += 1
                n_events_total += int(eta0_sel.size)

            # Process requested mechanisms for this transect
            if include_trans:
                mw = (w_trans if typ_l == "any" else 1.0)
                process_file(os.path.join(transects_dir, f"tr_{int(tid):04d}_trans_mcs.csv"), mix_weight=mw)

            if include_rot:
                mw = (w_rot if typ_l == "any" else 1.0)
                process_file(os.path.join(transects_dir, f"tr_{int(tid):04d}_rot_mcs.csv"), mix_weight=mw)
        # -----------------------
        # Finalize per-coast outputs
        # -----------------------
        out_rows = []
        for j in range(n_coast):
            if R_store[j]:
                Rj = np.concatenate(R_store[j]).astype(float)
                Wj = np.concatenate(W_store[j]).astype(float)
            else:
                Rj = np.array([], dtype=float)
                Wj = np.array([], dtype=float)

            # Annual rate of experiencing at least "coast_eta_min" runup
            thr = float(coast_eta_min)
            lam_thr = float(Wj[Rj >= thr].sum()) if Rj.size else 0.0

            row_out = dict(
                coast_id=int(coast_ids[j]), # unique ID of coastal point (from index.csv)
                x=int(x_coast[j]), # x coordinate of coastal point
                y=int(y_coast[j]), # y coordinate of coastal point
                lambda_RunUp_ge_thr_1yr=round(lam_thr,4), # annual rate of experiencing runup >= coast_eta_min at this coastal point
            )

            # Design runup for tsunami return periods
            for T in tsu_rps:
                row_out[f"RunUp_{int(T)}yr"] = design_runup_from_weighted_events(Rj, Wj, T_years=float(T))

            # Optional distribution summaries (sanity checks)
            row_out["RunUp_p50"] = round(weighted_percentile(Rj, Wj, 50.0),3) if Rj.size else np.nan
            row_out["RunUp_p90"] = round(weighted_percentile(Rj, Wj, 90.0),3) if Rj.size else np.nan
            row_out["RunUp_p99"] = round(weighted_percentile(Rj, Wj, 99.0),3) if Rj.size else np.nan

            out_rows.append(row_out)

        out = pd.DataFrame(out_rows)
        out_csv = os.path.join(coast_dir, f"coast_runup_eqRP{int(eq_rp_yr)}_{typ_l}.csv")
        out.to_csv(out_csv, index=False)

        print("[OK] wrote", out_csv)
        print(f"[INFO] type={typ_l} used_files={n_files_used} events_total={n_events_total} coast_points={n_coast}")

    print("[DONE] coastal runup outputs in:", coast_dir)


if __name__ == "__main__":
    main()


