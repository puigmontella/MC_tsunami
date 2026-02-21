#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import glob
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utilities.config import load_config, abspath


# -----------------------------
# Helpers
# -----------------------------
def _get(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _script_path(name: str) -> str:
    return os.path.join(ROOT, "Python", name)


def _run(cmd, dry_run=False):
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(f"[FAIL] Command failed (exit={p.returncode}): {' '.join(cmd)}")
    return p.returncode


def _count_glob(pattern: str) -> int:
    return len(glob.glob(pattern))


def _exists(path: str) -> bool:
    return path is not None and os.path.exists(path)


def _parse_steps(s: str):
    s = (s or "").strip().lower()
    if s in ("all", "everything"):
        return ["0", "1", "2", "3", "4r", "4t", "4f", "5", "6", "7"]
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok:
            out.append(tok)
    return out


def _parse_csv_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_int_list(s: str):
    return [int(x) for x in _parse_csv_list(s)]


def _infer_smax_from_transects(transects_dir: str) -> float | None:
    """
    Infer transect length (s_max) from existing transect CSVs produced by step 2.
    Looks for column names like s_m / dist_m / distance_m / s and returns max(s).
    """
    pat = os.path.join(transects_dir, "tr_*.csv")
    files = sorted(glob.glob(pat))

    # filter out non-transect files
    clean = []
    for f in files:
        b = os.path.basename(f)
        if b == "index.csv":
            continue
        if "_mcs" in b:
            continue
        if "transects_over" in b:
            continue
        clean.append(f)
    files = clean

    if not files:
        return None

    # try a couple files (in case first is odd)
    candidates = files[:5]
    for f in candidates:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        for key in ["s_m", "dist_m", "distance_m", "s"]:
            if key in cols:
                v = pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(float)
                if np.isfinite(v).any():
                    return float(np.nanmax(v))
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Run full MC tsunami pipeline (Grilli translational + rotational).")

    ap.add_argument("--config", required=True, help="Path to cases/config.json")
    ap.add_argument("--steps", default="1,2,3,4r,4t,5,6,7",
                    help="Comma list: 0,1,2,3,4r,4t,4f,5,6,7 or 'all'. Default: 1..7 (no step0).")

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run steps even if outputs exist")

    # Common MC options
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--n-trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-fs-checks", action="store_true")

    # RP ladder options (passed to steps 4*)
    ap.add_argument("--rp-min", type=float, default=100.0)
    ap.add_argument("--rp-max", type=float, default=750.0)
    ap.add_argument("--rp-step", type=float, default=10.0)
    ap.add_argument("--rp-cap", type=float, default=750.0)

    # Step 0 (optional contour extraction)
    ap.add_argument("--do-step0", action="store_true",
                    help="Allow step 0 to run when '0' is in --steps (still optional by default).")
    ap.add_argument("--contour-level", type=float, default=0.0)
    ap.add_argument("--contour-method", type=str, default="auto", choices=["auto", "contour0", "mask"])
    ap.add_argument("--contour-min-length-m", type=float, default=0.0)
    ap.add_argument("--contour-out-shp", default=None,
                    help="Override contour output shapefile (if step 0 is used).")

    # Step 1 (geomorphology) overrides
    ap.add_argument("--depth-cut", type=float, default=-2500.0)
    ap.add_argument("--make-geom-plots", action="store_true")

    # Step 3 (hazard fitting) override:
    # If not provided, pipeline will infer it from transect CSVs produced in step 2.
    ap.add_argument("--s-max", type=float, default=None,
                    help="Override transect length for 3_loading_PGA.py. If omitted, inferred from transect CSVs.")

    # Step 5 options
    ap.add_argument("--rps", default="100,200,475,750")
    ap.add_argument("--eta-min", type=float, default=0.5)

    # Step 6 options
    ap.add_argument("--map-types", default="any", help="Comma list for --types in 6_maps.py (e.g. any,rot,trans)")
    ap.add_argument("--eta-pct", type=int, default=99)

    args = ap.parse_args()

    cfg = load_config(args.config)
    py = sys.executable

    # Resolve main dirs from config (with safe fallbacks)
    transects_dir = abspath(cfg, _get(cfg, ["outputs", "transects_dir"], "outputs/transects"))
    rasters_dir = abspath(cfg, _get(cfg, ["outputs", "rasters_dir"], "outputs/rasters"))
    out_root = os.path.dirname(transects_dir.rstrip("/"))
    maps_dir = abspath(cfg, _get(cfg, ["outputs", "maps_dir"], os.path.join(out_root, "maps")))
    summary_csv = abspath(cfg, _get(cfg, ["outputs", "summary_csv"], os.path.join(out_root, "summary_eq_center_all_rp.csv")))
    hazard_dir = abspath(cfg, _get(cfg, ["outputs", "hazard_dir"], os.path.join(transects_dir, "hazard_fits")))
    hazard_index = os.path.join(hazard_dir, "hazard_fit_index.csv")

    steps = _parse_steps(args.steps)

    print("[PIPE] config:", abspath(cfg, args.config))
    print("[PIPE] rasters_dir:", rasters_dir)
    print("[PIPE] transects_dir:", transects_dir)
    print("[PIPE] hazard_dir:", hazard_dir)
    print("[PIPE] hazard index:", hazard_index)
    print("[PIPE] summary_csv:", summary_csv)
    print("[PIPE] maps_dir:", maps_dir)
    print("[PIPE] steps:", steps)

    os.makedirs(rasters_dir, exist_ok=True)
    os.makedirs(transects_dir, exist_ok=True)
    os.makedirs(hazard_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    # -----------------------------
    # Step 0: contour extraction (optional)
    # -----------------------------
    def step0():
        if not args.do_step0:
            print("[SKIP] step 0: not enabled (use --do-step0 and include '0' in --steps)")
            return

        bathy_tif = abspath(cfg, _get(cfg, ["paths", "bathy_tif"], None))
        if not _exists(bathy_tif):
            raise SystemExit(f"[FAIL] step 0: bathy_tif missing: {bathy_tif}")

        out_shp = args.contour_out_shp
        if out_shp is None:
            out_shp = _get(cfg, ["outputs", "contour_shp"], None) or _get(cfg, ["paths", "contour_shp"], None)
            if out_shp is None:
                out_shp = os.path.join(out_root, "contours", "contour0.shp")
        out_shp = abspath(cfg, out_shp)
        os.makedirs(os.path.dirname(out_shp), exist_ok=True)

        # NOTE: step0 script does NOT take --config
        cmd = [
            py, _script_path("0_make_ContourFromBathymetry.py"),
            "--bathy-tif", bathy_tif,
            "--out-shp", out_shp,
            "--level", str(args.contour_level),
            "--method", str(args.contour_method),
        ]
        if args.contour_min_length_m and args.contour_min_length_m > 0:
            cmd += ["--min-length-m", str(args.contour_min_length_m)]

        _run(cmd, args.dry_run)

    # -----------------------------
    # Step 1: geomorphology (creates slope_deg.tif etc)
    # -----------------------------
    def step1():
        out_slope = os.path.join(rasters_dir, "slope_deg.tif")
        if _exists(out_slope) and (not args.force):
            print("[SKIP] step 1: slope already exists:", out_slope)
            return

        cmd = [
            py, _script_path("1_geomorphology.py"),
            "--config", args.config,
            "--depth-cut", str(args.depth_cut),
        ]
        if args.make_geom_plots:
            cmd += ["--make-plots"]
        _run(cmd, args.dry_run)

        if not args.dry_run and not _exists(out_slope):
            raise SystemExit(f"[FAIL] step 1 did not produce: {out_slope}")

    # -----------------------------
    # Step 2: make transects (needs rasters)
    # -----------------------------
    def step2():
        n_tr = _count_glob(os.path.join(transects_dir, "tr_*.csv"))
        if n_tr > 0 and (not args.force):
            print(f"[SKIP] step 2: found {n_tr} existing transect CSVs in {transects_dir}")
            return

        need = os.path.join(rasters_dir, "slope_deg.tif")
        if not _exists(need):
            raise SystemExit(f"[FAIL] step 2 needs missing raster (run step 1 first): {need}")

        cmd = [py, _script_path("2_transect_maker.py"), "--config", args.config]
        _run(cmd, args.dry_run)

    # -----------------------------
    # Step 3: load PGA + fit hazard along transects
    # -----------------------------
    def step3():
        if _exists(hazard_index) and (not args.force):
            print("[SKIP] step 3: hazard_fit_index already exists:", hazard_index)
            return

        # IMPORTANT FIX:
        # 3_loading_PGA.py needs transect length if index.csv doesn't contain L_m/s_max_m.
        # If user didn't pass --s-max, infer it from the transect CSVs created in step2.
        smax = args.s_max
        if smax is None:
            smax = _infer_smax_from_transects(transects_dir)
            if smax is None:
                raise SystemExit(
                    "[FAIL] step 3: could not infer transect length from transect CSVs.\n"
                    "       Provide --s-max (e.g. --s-max 10000) or add L_m / s_max_m to transects/index.csv."
                )
            # nice round (optional): keep as-is but print
            print(f"[PIPE] step 3 inferred --s-max = {smax:.3f} m from transect CSVs")

        cmd = [
            py, _script_path("3_loading_PGA.py"),
            "--config", args.config,
            "--n-jobs", str(args.n_jobs),
            "--s-max", str(float(smax)),
        ]
        _run(cmd, args.dry_run)

        if not args.dry_run and not _exists(hazard_index):
            raise SystemExit(f"[FAIL] step 3 did not produce: {hazard_index}")

    # -----------------------------
    # Step 4r: rotational MC (Grilli)
    # -----------------------------
    def step4r():
        out_pat = os.path.join(transects_dir, "tr_*_grilli_rot_mcs.csv")
        n_out = _count_glob(out_pat)
        if n_out > 0 and (not args.force):
            print(f"[SKIP] step 4r: found {n_out} rotational MC CSVs")
            return

        if not _exists(hazard_index):
            raise SystemExit("[FAIL] step 4r needs hazard fits (run step 3 first).")

        cmd = [
            py, _script_path("4_rotational_mc.py"),
            "--config", args.config,
            "--n-trials", str(args.n_trials),
            "--n-jobs", str(args.n_jobs),
            "--seed", str(args.seed),
            "--rp-min", str(args.rp_min),
            "--rp-max", str(args.rp_max),
            "--rp-step", str(args.rp_step),
            "--rp-cap", str(args.rp_cap),
        ]
        if args.keep_fs_checks:
            cmd += ["--keep-fs-checks"]

        _run(cmd, args.dry_run)

    # -----------------------------
    # Step 4t: translational MC (Grilli)
    # -----------------------------
    def step4t():
        out_pat = os.path.join(transects_dir, "tr_*_grilli_mcs.csv")
        n_out = _count_glob(out_pat)
        if n_out > 0 and (not args.force):
            print(f"[SKIP] step 4t: found {n_out} translational MC CSVs")
            return

        if not _exists(hazard_index):
            raise SystemExit("[FAIL] step 4t needs hazard fits (run step 3 first).")

        cmd = [
            py, _script_path("4_translational_mc.py"),
            "--config", args.config,
            "--n-trials", str(args.n_trials),
            "--n-jobs", str(args.n_jobs),
            "--seed", str(args.seed),
            "--rp-min", str(args.rp_min),
            "--rp-max", str(args.rp_max),
            "--rp-step", str(args.rp_step),
            "--rp-cap", str(args.rp_cap),
        ]
        if args.keep_fs_checks:
            cmd += ["--keep-fs-checks"]

        _run(cmd, args.dry_run)

    # -----------------------------
    # Step 4f: optional regional filter (if you want it)
    # -----------------------------
    def step4f():
        script = _script_path("4_regional_filter.py")
        if not os.path.exists(script):
            print("[SKIP] step 4f: 4_regional_filter.py not found")
            return
        cmd = [py, script, "--config", args.config]
        _run(cmd, args.dry_run)

    # -----------------------------
    # Step 5: pf curves from RPmin
    # -----------------------------
    def step5():
        if _exists(summary_csv) and (not args.force):
            print("[SKIP] step 5: summary already exists:", summary_csv)
            return

        cmd = [
            py, _script_path("5_pf_from_rpmin.py"),
            "--config", args.config,
            "--rps", ",".join(_parse_csv_list(args.rps)),
            "--eta-min", str(args.eta_min),
        ]
        _run(cmd, args.dry_run)

        if not args.dry_run and not _exists(summary_csv):
            print("[WARN] step 5 finished but summary_csv not found at expected path:")
            print("       expected:", summary_csv)
            print("       (check script output for the actual written file)")

    # -----------------------------
    # Step 6: maps
    # -----------------------------
    def step6():
        cmd1 = [
            py, _script_path("6_maps.py"),
            "--config", args.config,
            "--what", "PfT",
            "--types", ",".join(_parse_csv_list(args.map_types)),
            "--rps", ",".join(_parse_csv_list(args.rps)),
        ]
        _run(cmd1, args.dry_run)

        cmd2 = [
            py, _script_path("6_maps.py"),
            "--config", args.config,
            "--what", "eta0_tsu",
            "--eta-pct", str(args.eta_pct),
            "--types", ",".join(_parse_csv_list(args.map_types)),
            "--rps", ",".join(_parse_csv_list(args.rps)),
        ]
        _run(cmd2, args.dry_run)

    # -----------------------------
    # Step 7: lightweight sanity checks (writes a txt report)
    # -----------------------------
    def step7():
        if not _exists(summary_csv):
            print("[SKIP] step 7: summary_csv missing:", summary_csv)
            return

        df = pd.read_csv(summary_csv)
        rps = _parse_int_list(args.rps)

        rep_lines = []
        rep_lines.append(f"summary_csv: {summary_csv}")
        rep_lines.append(f"n_transects: {len(df)}")
        rep_lines.append(f"rps: {rps}")
        rep_lines.append("")

        def _col(name):
            return pd.to_numeric(df.get(name, np.nan), errors="coerce").to_numpy(float)

        # mixture weights from trial counts (more general than hard-coded 0.5)
        Nrot = _col("n_trials_rot")
        Ntr  = _col("n_trials_trans")
        W = Nrot + Ntr
        wrot = np.where(W > 0, Nrot / W, np.nan)
        wtr  = np.where(W > 0, Ntr  / W, np.nan)

        for rp in rps:
            PfA = _col(f"Pf_{rp}_any")
            PfR = _col(f"Pf_{rp}_rot")
            PfT = _col(f"Pf_{rp}_trans")

            m = np.isfinite(PfA) & np.isfinite(PfR) & np.isfinite(PfT) & np.isfinite(wrot) & np.isfinite(wtr)
            if m.any():
                mix = wrot * PfR + wtr * PfT
                err = float(np.nanmax(np.abs(PfA[m] - mix[m])))
                rep_lines.append(f"RP={rp}: max|Pf_any - (wrot*rot + wtr*trans)| = {err:.3g}")
            else:
                rep_lines.append(f"RP={rp}: Pf_any/rot/trans finite rows = 0")

            PfTA = _col(f"PfT_{rp}_any")
            if np.isfinite(PfTA).any() and np.isfinite(PfA).any():
                rep_lines.append(f"RP={rp}: PfT_any <= Pf_any : {bool(np.nanmax(PfTA - PfA) <= 1e-12)}")

        # Monotonicity in RP
        A = np.vstack([_col(f"Pf_{rp}_any") for rp in rps]).T
        mrow = np.all(np.isfinite(A), axis=1)
        viol = int(np.sum(np.any(np.diff(A[mrow], axis=1) < -1e-12, axis=1))) if np.any(mrow) else 0
        rep_lines.append("")
        rep_lines.append(f"Pf_any monotonic violations: {viol} / {int(np.sum(mrow))}")

        report_path = os.path.join(out_root, "sanity_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rep_lines) + "\n")
        print("[OK] wrote sanity report:", report_path)

    # -----------------------------
    # Execute requested steps
    # -----------------------------
    dispatch = {
        "0": step0,
        "1": step1,
        "2": step2,
        "3": step3,
        "4r": step4r,
        "4t": step4t,
        "4f": step4f,
        "5": step5,
        "6": step6,
        "7": step7,
    }

    for st in steps:
        if st not in dispatch:
            raise SystemExit(f"[FAIL] Unknown step '{st}'. Valid: {list(dispatch.keys())}")
        print(f"\n[PIPE] ===== step {st} =====")
        dispatch[st]()

    print("\n[PIPE] DONE")


if __name__ == "__main__":
    main()