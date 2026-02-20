#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess


def project_root():
    """
    Locate repo root so:
      - imports work (utilities/)
      - scripts can be called as Python/<script>.py
    """
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, "..")) if os.path.basename(here) == "Python" else here
    if not os.path.isdir(os.path.join(root, "utilities")):
        raise SystemExit(f"Cannot find project root with utilities/. Started from: {here}")
    return root


ROOT = project_root()
PYDIR = os.path.join(ROOT, "Python")


def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)


def abs_from_cfg(cfg_path, p):
    """
    Resolve a path in the most forgiving way:
      1) absolute path => keep
      2) relative to config folder
      3) relative to repo root
    """
    if p is None:
        return None
    if os.path.isabs(p):
        return p

    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    cand1 = os.path.join(cfg_dir, p)
    if os.path.exists(cand1):
        return os.path.abspath(cand1)

    cand2 = os.path.join(ROOT, p)
    return os.path.abspath(cand2)


def run(cmd, env):
    print("\n>>> " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def main():
    ap = argparse.ArgumentParser(description="Run the MC_tsunami pipeline using a case config.json")
    ap.add_argument("--config", required=True, help="Path to config.json (e.g. cases/config.json)")

    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--to-step", type=int, default=8)

    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for MC + PGA steps")
    ap.add_argument("--regional", action="store_true", help="Apply the regional filter after MC runs")

    ap.add_argument("--n-trials-trans", type=int, default=5000)
    ap.add_argument("--n-trials-rot", type=int, default=5000)

    ap.add_argument("--make-plots", action="store_true", help="Step 1: write diagnostic PNGs")
    ap.add_argument("--plot-overview", action="store_true", help="Step 2: overview PNG")

    ap.add_argument("--make-coastline", action="store_true",
                    help="Force Step 0: generate coastline from bathy (even if config says source=file)")
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = load_cfg(cfg_path)

    py = sys.executable

    # Ensure `from utilities...` always works no matter where you run from
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    # Read a few config paths so Step 0 can decide what to do
    bathy_tif = abs_from_cfg(cfg_path, cfg["paths"]["bathy_tif"])

    coast_cfg = cfg["paths"].get("coastline", {})
    coast_source = str(coast_cfg.get("source", "file")).lower()
    coast_shp = abs_from_cfg(cfg_path, coast_cfg.get("shp")) if coast_cfg.get("shp") else None
    coast_generated = abs_from_cfg(cfg_path, coast_cfg.get("generated_out")) if coast_cfg.get("generated_out") else None

    trans_params = abs_from_cfg(cfg_path, cfg["paths"]["parameters"]["trans"]) if "parameters" in cfg["paths"] else None
    rot_params   = abs_from_cfg(cfg_path, cfg["paths"]["parameters"]["rot"])   if "parameters" in cfg["paths"] else None

    def script(name):
        return os.path.join(PYDIR, name)

    # --------------------------
    # Define steps (simple list)
    # --------------------------
    steps = {}

    # 0) optional: make coastline from bathy
    #    (only runs if config says source=bathy OR you pass --make-coastline)
    steps[0] = [
        py, script("0_make_ContourFromBathymetry.py"),
        "--bathy-tif", bathy_tif,
        "--out-shp", coast_generated if coast_generated else "",
        "--method", "mask",
    ]

    # 1) bathy -> slope/roughness/class rasters
    steps[1] = [py, script("1_geomorphology.py"), "--config", cfg_path]
    if args.make_plots:
        steps[1].append("--make-plots")

    # 2) coastline + rasters -> transects
    steps[2] = [py, script("2_transect_maker.py"), "--config", cfg_path]
    if args.plot_overview:
        steps[2].append("--plot-overview")

    # 3) translational MC
    steps[3] = [
        py, script("3_translational_mc.py"),
        "--config", cfg_path,
        "--params", trans_params if trans_params else "",
        "--n-trials", str(args.n_trials_trans),
        "--n-jobs", str(args.n_jobs),
    ]

    # 4) rotational MC
    steps[4] = [
        py, script("3_rotational_mc.py"),
        "--config", cfg_path,
        "--params", rot_params if rot_params else "",
        "--n-trials", str(args.n_trials_rot),
        "--n-jobs", str(args.n_jobs),
    ]

    # 5) regional filter (trans outputs)
    steps[5] = [py, script("4_regional_filter.py"), "--config", cfg_path, "--mode", "trans"]

    # 6) regional filter (rot outputs)
    steps[6] = [py, script("4_regional_filter.py"), "--config", cfg_path, "--mode", "rot"]

    # 7) sample PGA + compute Pf / eta tails -> summary CSV
    steps[7] = [py, script("5_loading_PGA.py"), "--config", cfg_path, "--n-jobs", str(args.n_jobs)]
    # 8) maps
    steps[8] = [py, script("6_maps.py"), "--config", cfg_path]


    steps[9] = [py, script("7_sanity_checks.py"), "--config", cfg_path]
    if args.make_plots:
        steps[9].append("--plots")


    # --------------------------
    # Decide run order
    # --------------------------
    order = []

    # coastline: do we need to generate it?
    need_make_coast = args.make_coastline or (coast_source == "bathy")
    if need_make_coast:
        if not coast_generated:
            raise SystemExit("Config says coastline from bathy, but paths.coastline.generated_out is missing.")
        order.append(0)
    else:
        # using shapefile
        if not coast_shp or (not os.path.exists(coast_shp)):
            raise SystemExit("Config says coastline from file, but paths.coastline.shp is missing or not found.")

    order += [1, 2, 3, 4]
    if args.regional:
        order += [5, 6]
    order += [7, 8]

    # --------------------------
    # Run
    # --------------------------
    for s in order:
        if s < args.from_step or s > args.to_step:
            continue
        cmd = [c for c in steps[s] if c != ""]
        run(cmd, env)

    print("\n[DONE] Pipeline finished.")


if __name__ == "__main__":
    main()
