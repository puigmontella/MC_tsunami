#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

# project root on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utilities.config import load_config, abspath


def _as_float(x):
    return pd.to_numeric(x, errors="coerce").to_numpy(float)


def _fail(msg, errors):
    errors.append(msg)
    print("[FAIL]", msg)


def _warn(msg):
    print("[WARN]", msg)


def _ok(msg):
    print("[ OK ]", msg)


def _safe_read_csv(path):
    """Read CSV robustly; return None if empty/broken."""
    try:
        if os.path.exists(path) and os.path.getsize(path) == 0:
            return None
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def check_summary(df, rps, types=("rot", "trans", "any"), atol=1e-12):
    errors = []

    # ---- required columns
    need = ["transect_id"]
    for rp in rps:
        for pref in ("Pf", "PfT"):
            for ty in types:
                need.append(f"{pref}_{rp}_{ty}")

    missing = [c for c in need if c not in df.columns]
    if missing:
        _fail(f"summary missing columns (first 20): {missing[:20]}", errors)
        return errors

    # ---- PfT <= Pf
    for rp in rps:
        for ty in types:
            Pf  = _as_float(df[f"Pf_{rp}_{ty}"])
            PfT = _as_float(df[f"PfT_{rp}_{ty}"])
            m = np.isfinite(Pf) & np.isfinite(PfT)
            if m.any():
                mx = np.nanmax(PfT[m] - Pf[m])
                if mx > atol:
                    _fail(f"PfT > Pf for rp={rp} type={ty}: max(PfT-Pf)={mx}", errors)
    _ok("PfT <= Pf passed")

    # ---- monotonicity in RP
    for pref in ("Pf", "PfT"):
        for ty in types:
            A = np.vstack([_as_float(df[f"{pref}_{rp}_{ty}"]) for rp in rps]).T
            m = np.all(np.isfinite(A), axis=1)
            if not m.any():
                _warn(f"no finite rows for monotonic check: {pref}_{ty}")
                continue
            viol = int(np.sum(np.any(np.diff(A[m], axis=1) < -atol, axis=1)))
            if viol:
                _fail(f"monotonic violations for {pref}_{ty}: {viol}/{int(np.sum(m))}", errors)
    _ok("Monotonic-in-RP passed")

    # ---- mixture identity (if we have trial counts)
    have_counts = ("n_trials_rot" in df.columns) and ("n_trials_trans" in df.columns)
    if not have_counts:
        _warn("n_trials_rot/n_trials_trans missing -> skipping mixture weighted check")
        return errors

    Nrot = _as_float(df["n_trials_rot"])
    Ntr  = _as_float(df["n_trials_trans"])
    W = Nrot + Ntr
    wrot = np.where(W > 0, Nrot / W, np.nan)
    wtr  = np.where(W > 0, Ntr  / W, np.nan)

    for rp in rps:
        for pref in ("Pf", "PfT"):
            anyv = _as_float(df[f"{pref}_{rp}_any"])
            rotv = _as_float(df[f"{pref}_{rp}_rot"])
            trnv = _as_float(df[f"{pref}_{rp}_trans"])
            mix = wrot * rotv + wtr * trnv

            m = np.isfinite(anyv) & np.isfinite(mix)
            if m.any():
                err = float(np.nanmax(np.abs(anyv[m] - mix[m])))
                if err > 1e-10:  # should be ~1e-16 for you
                    _fail(f"mixture identity fails {pref} rp={rp}: max|any-mix|={err}", errors)

            # bounds check
            m2 = np.isfinite(anyv) & np.isfinite(rotv) & np.isfinite(trnv)
            if m2.any():
                lo = np.minimum(rotv, trnv)
                hi = np.maximum(rotv, trnv)
                bad = int(np.sum((anyv[m2] < lo[m2] - atol) | (anyv[m2] > hi[m2] + atol)))
                if bad:
                    _fail(f"mixture bounds fail {pref} rp={rp}: bad={bad}/{int(np.sum(m2))}", errors)

    _ok("Mixture identity + bounds passed")
    return errors


def check_duplicates(transects_dir):
    # just warn if both *_REGIONAL and non-REGIONAL exist
    rot_files = sorted(glob.glob(os.path.join(transects_dir, "tr_*_grilli_rot_mcs*.csv")))
    if not rot_files:
        _warn("no rotational MC files found in transects_dir")
        return

    # group by base name without _REGIONAL
    groups = {}
    for f in rot_files:
        b = os.path.basename(f)
        key = b.replace("_REGIONAL", "")
        groups.setdefault(key, []).append(b)

    dups = {k: v for k, v in groups.items() if len(v) > 1}
    if dups:
        _warn(f"found duplicate rotational outputs (REGIONAL vs non): {len(dups)} transects (ok if your scripts choose one pattern)")


def main():
    ap = argparse.ArgumentParser(description="Step 7 sanity checks for MC tsunami pipeline outputs.")
    ap.add_argument("--config", required=True, help="cases/config.json")
    ap.add_argument("--rps", default="100,200,475,750", help="comma list of RPs to test")
    ap.add_argument("--atol", type=float, default=1e-12, help="tolerance for comparisons")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Your config DOES have outputs.transects_dir (we rely on that)
    if "outputs" not in cfg or "transects_dir" not in cfg["outputs"]:
        print("[FAIL] config missing outputs.transects_dir")
        raise SystemExit(2)

    transects_dir = abspath(cfg, cfg["outputs"]["transects_dir"])

    # BASIC + robust: out_dir is parent of transects_dir
    out_dir = os.path.dirname(os.path.abspath(transects_dir))

    summary_path = os.path.join(out_dir, "summary_eq_center_all_rp.csv")

    rps = [int(x) for x in args.rps.split(",") if x.strip()]

    print("[INFO] transects_dir:", transects_dir)
    print("[INFO] out_dir      :", out_dir)
    print("[INFO] summary      :", summary_path)
    print("[INFO] rps          :", rps)

    if not os.path.exists(summary_path):
        print("[FAIL] missing summary:", summary_path)
        raise SystemExit(2)

    df = _safe_read_csv(summary_path)
    if df is None or len(df) == 0:
        print("[FAIL] summary is empty/unreadable:", summary_path)
        raise SystemExit(2)

    errors = []
    errors += check_summary(df, rps=rps, atol=args.atol)
    check_duplicates(transects_dir)

    if errors:
        print("\n[DONE] sanity check: FAILED with", len(errors), "errors")
        raise SystemExit(2)

    print("\n[DONE] sanity check: PASSED")
    raise SystemExit(0)


if __name__ == "__main__":
    main()