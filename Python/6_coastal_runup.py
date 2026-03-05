#!/usr/bin/env python3
"""
Coastal runup proxy from Monte Carlo source trials.

Reads:
  - <transects_dir>/index.csv                      (coastal points + normals)
  - <transects_dir>/tr_XXXX_trans_mcs.csv         (trans)
  - <transects_dir>/tr_XXXX_rot_mcs.csv           (rot)
  - <areas_dir>/ar_XXXX_trans_mcs.csv             (trans)
  - <areas_dir>/ar_XXXX_rot_mcs.csv               (rot)

Logic (for a chosen earthquake return period RP_eq):
  - Select only trials that:
        RPmin_yr <= RP_eq
        eta0_m   >= eta_min_src   (eta_min_src from config: pga_loading.eta_min_m)

  - Spread runup to all coastal receivers using angular Gaussian:
        W = exp(-0.5*(dtheta/sigma)^2)
    with hard cutoff |dtheta| > max_deg -> W=0
    and optional jitter on main direction (uniform in [-jitter, +jitter]).

  - Compute "design runup" for tsunami return periods T:
        find R_T such that sum_{events with R >= R_T} w_event >= 1/T

Outputs:
  - <coast_dir>/coast_runup_eqRP{RP}_{type}_{source}.csv
    where source in {transects, areas, both}

Notes
-----
- --source-level supports: transects, areas, both, all
- --theta0-mode:
      physics  : transects use outward normal; areas use azimuth_deg if present (recommended)
      to_coast : old behavior (theta0 = source -> nearest/own coast anchor)
- when source-level=both, we ALSO write legacy name:
      coast_runup_eqRP{RP}_{type}.csv
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd

from utilities.stats import weighted_percentile


# -------------------------
# small helpers
# -------------------------
def count_data_rows_csv(path):
    """Fast-ish row count without pandas (counts lines minus header)."""
    try:
        with open(path, "rb") as f:
            n = 0
            for _ in f:
                n += 1
        return max(0, n - 1)
    except Exception:
        return 0


def design_runup_from_weighted_events(R, w, T_years):
    """
    Return design runup R_T such that weighted annual exceedance rate >= 1/T.
    Here w is an annual rate per event.
    Returns NaN if the target exceedance rate is not reachable.
    """
    R = np.asarray(R, float)
    w = np.asarray(w, float)

    m = np.isfinite(R) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return np.nan

    Rm = R[m]
    wm = w[m]

    idx = np.argsort(Rm)[::-1]
    R_sorted = Rm[idx]
    w_sorted = wm[idx]

    cum = np.cumsum(w_sorted)
    target = 1.0 / float(T_years)

    if cum.size == 0 or cum[-1] < target:
        return np.nan

    j = int(np.searchsorted(cum, target, side="left"))
    j = min(max(j, 0), R_sorted.size - 1)
    return round(float(R_sorted[j]), 3)


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


def nearest_anchor_theta0(x_src, y_src, x_coast, y_coast):
    """
    For each source point, find nearest coast point and compute theta0
    (direction from source -> nearest coast anchor).
    Returns:
      theta0 (n,), x_anchor (n,), y_anchor (n,)
    """
    dx = x_coast[None, :] - x_src[:, None]
    dy = y_coast[None, :] - y_src[:, None]
    dd = dx * dx + dy * dy
    j = np.argmin(dd, axis=1)
    xa = x_coast[j]
    ya = y_coast[j]
    theta0 = np.arctan2((ya - y_src), (xa - x_src))
    return theta0, xa, ya


def compute_transect_segment_lengths(idx_df):
    """
    Build a mapping transect_id -> alongshore segment length (m) from coast_s_m.
    This approximates how much coastline each transect represents.
    """
    if "coast_s_m" not in idx_df.columns:
        tids = pd.to_numeric(idx_df["transect_id"], errors="coerce").to_numpy(int)
        return {int(t): 1.0 for t in tids}

    tmp = idx_df.copy()
    tmp["transect_id"] = pd.to_numeric(tmp["transect_id"], errors="coerce")
    tmp["coast_s_m"] = pd.to_numeric(tmp["coast_s_m"], errors="coerce")
    tmp = tmp.dropna(subset=["transect_id", "coast_s_m"]).sort_values("coast_s_m")

    tids = tmp["transect_id"].to_numpy(int)
    s = tmp["coast_s_m"].to_numpy(float)

    if len(s) < 2:
        return {int(t): 1.0 for t in tids}

    ds = np.empty_like(s)
    ds[0] = s[1] - s[0]
    ds[-1] = s[-1] - s[-2]
    if len(s) > 2:
        ds[1:-1] = 0.5 * (s[2:] - s[:-2])

    ds = np.where(np.isfinite(ds) & (ds > 0), ds, np.nanmedian(ds[np.isfinite(ds) & (ds > 0)]))
    ds = np.where(np.isfinite(ds) & (ds > 0), ds, 1.0)

    return {int(tids[i]): float(ds[i]) for i in range(len(tids))}


def azimuth_deg_to_theta_rad(az_deg):
    """
    Convert azimuth degrees clockwise from North (0=N,90=E) to atan2(dy,dx) radians
    (0=+x East, +pi/2=+y North).
      theta = deg2rad(90 - azimuth_deg)
    """
    return np.deg2rad(90.0 - np.asarray(az_deg, float))


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Coastal runup + design runup from MC trials.")
    ap.add_argument("--config", default="config.json", help="Path to case config.json")
    ap.add_argument("--types", default=None,
                    help="Comma list among: trans, rot, any. If omitted, uses coast_runup.json types.")
    ap.add_argument("--source-level", default="both",
                    help="One of: transects, areas, both, all. Default: both")
    ap.add_argument("--theta0-mode", default="physics", choices=["physics", "to_coast"],
                    help="Direction model for spreading: physics (recommended) or to_coast (old).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # --- Load case config ---
    if not os.path.exists(args.config):
        raise SystemExit(f"[FAIL] config not found: {args.config}")
    with open(args.config, "r") as f:
        cfg = json.load(f)

    transects_dir = cfg["outputs"]["transects_dir"]
    areas_dir = cfg["outputs"].get("areas_dir", "outputs/areas")
    coast_dir = cfg["outputs"]["coast_dir"]
    os.makedirs(coast_dir, exist_ok=True)

    eta_min_src = float(cfg["pga_loading"]["eta_min_m"])

    runup_params_path = cfg["paths"]["parameters"]["coast_runup"]
    if not os.path.exists(runup_params_path):
        raise SystemExit(f"[FAIL] Missing coast_runup params JSON: {runup_params_path}")
    with open(runup_params_path, "r") as f:
        runup_params = json.load(f)

    eq_rp_yr = float(runup_params["eq_rp_yr"])
    tsu_rps = [int(x) for x in runup_params["tsu_rps_yr"]]
    coast_eta_min = float(runup_params["coast_eta_min_m"])

    sigma = np.radians(float(runup_params["sigma_deg"]))
    max_a = np.radians(float(runup_params["max_deg"]))
    jitter = np.radians(float(runup_params["jitter_deg"]))

    w_rot = float(runup_params.get("w_rot", 0.5))
    w_trans = float(runup_params.get("w_trans", 0.5))

    # Types to run (CLI overrides JSON)
    if args.types is not None:
        types_to_run = [t.strip().lower() for t in str(args.types).split(",") if t.strip()]
        print("[INFO] types_to_run:", ",".join(types_to_run))
        print("[INFO] (types set from CLI --types)")
    else:
        types_to_run = [str(t).strip().lower() for t in runup_params.get("types", ["any"])]
        types_to_run = [("trans" if t == "tran" else t) for t in types_to_run]
        print("[INFO] types_to_run:", ",".join(types_to_run))
        print("[INFO] (types set from coast_runup.json)")

    # Source levels to run
    sl = str(args.source_level).strip().lower()
    if sl in ("transect", "transects", "tran"):
        src_levels = ["transects"]
    elif sl in ("area", "areas"):
        src_levels = ["areas"]
    elif sl == "both":
        src_levels = ["both"]
    elif sl == "all":
        src_levels = ["transects", "areas", "both"]
    else:
        raise SystemExit(f"[FAIL] unknown --source-level '{args.source_level}' (use transects/areas/both/all)")

    print("[INFO] source_level:", sl)
    if sl == "all":
        print("[INFO] source_levels_to_run:", ",".join(src_levels))
    print("[INFO] theta0_mode:", args.theta0_mode)

    rng = np.random.default_rng()

    # -----------------------
    # Load coastal index points
    # -----------------------
    index_csv = os.path.join(transects_dir, "index.csv")
    if not os.path.exists(index_csv):
        raise SystemExit(f"[FAIL] Missing index.csv: {index_csv}")

    idx = pd.read_csv(index_csv).rename(columns={"id": "transect_id"})
    for c in ("transect_id", "x0", "y0", "nx", "ny"):
        if c not in idx.columns:
            raise SystemExit(f"[FAIL] index.csv missing required column: {c}")

    coast_ids = pd.to_numeric(idx["transect_id"], errors="coerce").to_numpy(int)
    x_coast = pd.to_numeric(idx["x0"], errors="coerce").to_numpy(float)
    y_coast = pd.to_numeric(idx["y0"], errors="coerce").to_numpy(float)

    n_coast = x_coast.size
    if n_coast < 2:
        raise SystemExit("Need at least 2 coastal points in index.csv for alongshore spreading.")

    # segment lengths for transect weighting (optional but helps make priors closer to areas)
    seglen_by_tid = compute_transect_segment_lengths(idx)

    # representative offshore length for transect "catchment area"
    L_off = float(cfg.get("transects", {}).get("max_len_m", 15000.0))
    A_tr_by_tid = {int(t): float(seglen_by_tid.get(int(t), 1.0) * L_off) for t in coast_ids}

    # Effective area per tile for areas (use stride^2 as "unique area" when tiles overlap)
    stride_m = float(cfg.get("areas", {}).get("stride_m", 1.0))
    A_area_tile = float(stride_m * stride_m)

    # quick file lists
    tr_trans_files = sorted(glob.glob(os.path.join(transects_dir, "tr_*_trans_mcs.csv")))
    tr_rot_files   = sorted(glob.glob(os.path.join(transects_dir, "tr_*_rot_mcs.csv")))
    ar_trans_files = sorted(glob.glob(os.path.join(areas_dir, "ar_*_trans_mcs.csv")))
    ar_rot_files   = sorted(glob.glob(os.path.join(areas_dir, "ar_*_rot_mcs.csv")))
    print(f"[INFO] areas: trans_files={len(ar_trans_files)} rot_files={len(ar_rot_files)}")
    print(f"[INFO] transects: trans_files={len(tr_trans_files)} rot_files={len(tr_rot_files)}")

    # -----------------------
    # Main loop over source levels
    # -----------------------
    for src_level in src_levels:
        include_transects = (src_level in ("transects", "both"))
        include_areas = (src_level in ("areas", "both"))

        for typ_l in types_to_run:
            if typ_l not in ("trans", "rot", "any"):
                print("[WARN] unknown type:", typ_l, "skipping.")
                continue

            has_trans = (len(tr_trans_files) + len(ar_trans_files) > 0)
            has_rot   = (len(tr_rot_files) + len(ar_rot_files) > 0)

            if typ_l == "trans":
                mech_weights = {"trans": 1.0}
                if not has_trans:
                    print(f"[WARN] no trans files found for source={src_level} -> writing zeros")
                    mech_weights = {"trans": 0.0}
            elif typ_l == "rot":
                mech_weights = {"rot": 1.0}
                if not has_rot:
                    print(f"[WARN] no rot files found for source={src_level} -> writing zeros")
                    mech_weights = {"rot": 0.0}
            else:
                wt = w_trans if has_trans else 0.0
                wr = w_rot if has_rot else 0.0
                s = wt + wr
                if s <= 0:
                    print(f"[WARN] no trans/rot files found for source={src_level} -> writing zeros")
                    mech_weights = {"trans": 0.0, "rot": 0.0}
                else:
                    mech_weights = {"trans": wt/s, "rot": wr/s}

            # -----------------------------------------
            # Precompute normalization DENOM and base_w
            # -----------------------------------------
            denom = 0.0

            if include_transects:
                for tid in coast_ids:
                    if mech_weights.get("trans", 0.0) > 0:
                        p = os.path.join(transects_dir, f"tr_{int(tid):04d}_trans_mcs.csv")
                        nacc = count_data_rows_csv(p)
                        denom += mech_weights["trans"] * A_tr_by_tid.get(int(tid), 1.0) * float(nacc)
                    if mech_weights.get("rot", 0.0) > 0:
                        p = os.path.join(transects_dir, f"tr_{int(tid):04d}_rot_mcs.csv")
                        nacc = count_data_rows_csv(p)
                        denom += mech_weights["rot"] * A_tr_by_tid.get(int(tid), 1.0) * float(nacc)

            if include_areas:
                if mech_weights.get("trans", 0.0) > 0:
                    for p in ar_trans_files:
                        nacc = count_data_rows_csv(p)
                        denom += mech_weights["trans"] * A_area_tile * float(nacc)
                if mech_weights.get("rot", 0.0) > 0:
                    for p in ar_rot_files:
                        nacc = count_data_rows_csv(p)
                        denom += mech_weights["rot"] * A_area_tile * float(nacc)

            if denom <= 0:
                base_w = 0.0
                print(f"[WARN] denom=0 for type={typ_l} source={src_level} -> writing zeros")
            else:
                base_w = (1.0 / float(eq_rp_yr)) / float(denom)
                print(f"[INFO] norm: type={typ_l} source={src_level} DENOM={denom:.1f}  base_w={base_w:.3e}")

            R_store = [[] for _ in range(n_coast)]
            W_store = [[] for _ in range(n_coast)]

            n_files_used = 0
            n_events_total = 0

            def spread_events(eta0_sel, x_src, y_src, theta0, w_event):
                nonlocal n_events_total
                if eta0_sel.size == 0 or w_event <= 0:
                    return

                if jitter > 0:
                    theta0 = theta0 + rng.uniform(-jitter, +jitter, size=theta0.shape)

                dx = x_coast[None, :] - x_src[:, None]
                dy = y_coast[None, :] - y_src[:, None]
                theta = np.arctan2(dy, dx)

                dtheta = np.arctan2(np.sin(theta - theta0[:, None]), np.cos(theta - theta0[:, None]))

                if sigma > 0:
                    W = np.exp(-0.5 * (dtheta / sigma) ** 2)
                else:
                    W = (np.abs(dtheta) <= max_a).astype(float)

                if max_a > 0:
                    W[np.abs(dtheta) > max_a] = 0.0

                R = (eta0_sel[:, None] * W).astype(np.float32)
                ww = np.full((eta0_sel.size,), float(w_event), dtype=np.float32)

                for j in range(n_coast):
                    R_store[j].append(R[:, j])
                    W_store[j].append(ww)

                n_events_total += int(eta0_sel.size)

            def process_mcs_file(mcs_path, kind, mech, idx_row=None, tid=None):
                nonlocal n_files_used

                if not os.path.exists(mcs_path) or base_w <= 0:
                    return

                df = pd.read_csv(mcs_path)
                if df is None or df.empty:
                    return

                if "RPmin_yr" not in df.columns or "eta0_m" not in df.columns:
                    return

                RPmin = pd.to_numeric(df["RPmin_yr"], errors="coerce").to_numpy(float)
                eta0 = pd.to_numeric(df["eta0_m"], errors="coerce").to_numpy(float)

                msel = (
                    np.isfinite(RPmin) & (RPmin <= eq_rp_yr) &
                    np.isfinite(eta0) & (eta0 >= eta_min_src)
                )
                if not np.any(msel):
                    return

                eta0_sel = eta0[msel]

                mix = float(mech_weights.get(mech, 0.0))
                if mix <= 0:
                    return

                if kind == "transect":
                    A_file = float(A_tr_by_tid.get(int(tid), 1.0))
                else:
                    A_file = float(A_area_tile)

                w_event = base_w * mix * A_file

                if kind == "transect":
                    if idx_row is None:
                        return
                    for col in ("s0_m", "s1_m"):
                        if col not in df.columns:
                            return

                    s0 = pd.to_numeric(df["s0_m"], errors="coerce").to_numpy(float)
                    s1 = pd.to_numeric(df["s1_m"], errors="coerce").to_numpy(float)
                    s_center = 0.5 * (s0[msel] + s1[msel])

                    x_src = np.empty_like(s_center, dtype=float)
                    y_src = np.empty_like(s_center, dtype=float)
                    for i in range(s_center.size):
                        xs, ys = infer_source_xy_from_index(idx_row, s_center[i])
                        x_src[i] = xs
                        y_src[i] = ys

                    if args.theta0_mode == "physics":
                        # physics-first: runout direction ~ transect outward normal
                        nx = float(idx_row["nx"])
                        ny = float(idx_row["ny"])
                        theta0_const = np.arctan2(ny, nx)
                        theta0 = np.full_like(eta0_sel, theta0_const, dtype=float)
                    else:
                        # old behavior: source -> its coastal anchor point
                        x0 = float(idx_row["x0"])
                        y0 = float(idx_row["y0"])
                        theta0 = np.arctan2((y0 - y_src), (x0 - x_src))

                    spread_events(eta0_sel, x_src, y_src, theta0, w_event)

                else:
                    for col in ("x_center_m", "y_center_m"):
                        if col not in df.columns:
                            return

                    x_src = pd.to_numeric(df["x_center_m"], errors="coerce").to_numpy(float)[msel]
                    y_src = pd.to_numeric(df["y_center_m"], errors="coerce").to_numpy(float)[msel]

                    mm = np.isfinite(x_src) & np.isfinite(y_src) & np.isfinite(eta0_sel)
                    if not np.any(mm):
                        return

                    x_src = x_src[mm]
                    y_src = y_src[mm]
                    eta0_sel = eta0_sel[mm]

                    if args.theta0_mode == "physics" and ("azimuth_deg" in df.columns):
                        az = pd.to_numeric(df["azimuth_deg"], errors="coerce").to_numpy(float)[msel]
                        az = az[mm]
                        good = np.isfinite(az)
                        if np.any(good):
                            theta0 = azimuth_deg_to_theta_rad(az)
                        else:
                            theta0, _, _ = nearest_anchor_theta0(x_src, y_src, x_coast, y_coast)
                    else:
                        theta0, _, _ = nearest_anchor_theta0(x_src, y_src, x_coast, y_coast)

                    spread_events(eta0_sel, x_src, y_src, theta0, w_event)

                n_files_used += 1

            # -----------------------
            # Process TRANSECTS
            # -----------------------
            if include_transects and base_w > 0:
                for _, row in idx.iterrows():
                    tid = int(row["transect_id"])
                    if mech_weights.get("trans", 0.0) > 0:
                        process_mcs_file(
                            os.path.join(transects_dir, f"tr_{tid:04d}_trans_mcs.csv"),
                            kind="transect", mech="trans", idx_row=row, tid=tid
                        )
                    if mech_weights.get("rot", 0.0) > 0:
                        process_mcs_file(
                            os.path.join(transects_dir, f"tr_{tid:04d}_rot_mcs.csv"),
                            kind="transect", mech="rot", idx_row=row, tid=tid
                        )

            # -----------------------
            # Process AREAS
            # -----------------------
            if include_areas and base_w > 0:
                if mech_weights.get("trans", 0.0) > 0:
                    for fcsv in ar_trans_files:
                        process_mcs_file(fcsv, kind="area", mech="trans")
                if mech_weights.get("rot", 0.0) > 0:
                    for fcsv in ar_rot_files:
                        process_mcs_file(fcsv, kind="area", mech="rot")

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

                lam_selected = float(Wj.sum()) if Wj.size else 0.0

                thr = float(coast_eta_min)
                lam_thr = float(Wj[Rj >= thr].sum()) if Rj.size else 0.0

                row_out = dict(
                    coast_id=int(coast_ids[j]),
                    x=int(x_coast[j]),
                    y=int(y_coast[j]),
                    lambda_selected_1yr=round(lam_selected, 6),
                    lambda_RunUp_ge_thr_1yr=round(lam_thr, 6),
                )

                for T in tsu_rps:
                    row_out[f"RunUp_{int(T)}yr"] = design_runup_from_weighted_events(Rj, Wj, T_years=float(T))

                row_out["RunUp_p50"] = round(weighted_percentile(Rj, Wj, 50.0), 3) if Rj.size else np.nan
                row_out["RunUp_p90"] = round(weighted_percentile(Rj, Wj, 90.0), 3) if Rj.size else np.nan
                row_out["RunUp_p99"] = round(weighted_percentile(Rj, Wj, 99.0), 3) if Rj.size else np.nan

                out_rows.append(row_out)

            out = pd.DataFrame(out_rows)

            out_csv = os.path.join(coast_dir, f"coast_runup_eqRP{int(eq_rp_yr)}_{typ_l}_{src_level}.csv")
            out.to_csv(out_csv, index=False)
            print("[OK] wrote", out_csv)
            print(f"[INFO] type={typ_l} source={src_level} used_files={n_files_used} events_total={n_events_total} coast_points={n_coast}")

            if src_level == "both":
                legacy_csv = os.path.join(coast_dir, f"coast_runup_eqRP{int(eq_rp_yr)}_{typ_l}.csv")
                out.to_csv(legacy_csv, index=False)
                if args.verbose:
                    print("[OK] wrote legacy", legacy_csv)

    print("[DONE] coastal runup outputs in:", coast_dir)


if __name__ == "__main__":
    main()