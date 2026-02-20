#!/usr/bin/env python3
"""
Step 7 — sanity checks for the whole pipeline (config-driven)

Checks:
  A) Inputs exist + CRS/bounds consistency (bathy, coastline, PGA rasters)
  B) Transects: index + per-transect CSV integrity + "offshore" direction test
  C) Depth consistency: depth column vs resampled bathy at transect points (RMSE)
  D) MC outputs: empty files, required columns, kcrit/eta0 sanity summaries (+ eta0 unit mismatch hint)
  E) Summary CSV: columns present, Pf in [0,1], monotonic with RP, PGA monotonic with RP
  F) Hazard sanity (if present): PfH ordering, RPH ~ 1/AERH, PfH threshold monotonicity
  G) Optional quick plots (histograms + Pf vs PGA scatter)

Exit codes:
  0 = OK (no errors; warnings allowed unless --fail-on-warn)
  1 = warnings only (and --fail-on-warn)
  2 = errors found
"""

import os
import sys
import json
import glob
import re
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio

# project root on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utilities.config import load_config, abspath  # noqa: E402

# geopandas optional (coastline checks)
try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

# matplotlib optional (plots)
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


# -------------------------
# Small reporting helpers
# -------------------------
@dataclass
class Report:
    ok: List[str] = field(default_factory=list)
    warn: List[str] = field(default_factory=list)
    err: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_ok(self, msg: str):
        self.ok.append(msg)

    def add_warn(self, msg: str):
        self.warn.append(msg)

    def add_err(self, msg: str):
        self.err.append(msg)

    def summary(self) -> str:
        return f"OK={len(self.ok)} | WARN={len(self.warn)} | ERR={len(self.err)}"


def _sort_rps(rps: List[str]) -> List[str]:
    def key(x):
        try:
            return int(str(x))
        except Exception:
            return 10**18

    return sorted([str(r) for r in rps], key=key)


def _infer_rps_from_summary_columns(cols: List[str]) -> List[str]:
    rps = set()
    for c in cols:
        if c.startswith("Pf_"):
            parts = c.split("_")
            if len(parts) >= 3:
                rps.add(parts[1])
        elif c.startswith("eta0_fail"):
            s = c[len("eta0_fail"):]
            rp = ""
            for ch in s:
                if ch.isdigit():
                    rp += ch
                else:
                    break
            if rp:
                rps.add(rp)
    return _sort_rps(list(rps))


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Read CSV but return None if empty/bad."""
    try:
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _quant(x: np.ndarray, qs=(0.05, 0.5, 0.95)) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    q = np.quantile(x, qs)
    return float(q[0]), float(q[1]), float(q[2])


# -------------------------
# Core checks
# -------------------------
def check_inputs(cfg: Dict[str, Any], rep: Report) -> Dict[str, Any]:
    paths = cfg.get("paths", {})
    outs = cfg.get("outputs", {})

    bathy = abspath(cfg, paths["bathy_tif"])
    rep.metrics["bathy_tif"] = bathy

    if not os.path.exists(bathy):
        rep.add_err(f"Missing bathy GeoTIFF: {bathy}")
        return {}

    # PGA rasters
    pga_rasters = paths.get("pga_rasters", {})
    pga_abs = {}
    for rp, p in pga_rasters.items():
        ap = abspath(cfg, p)
        pga_abs[str(rp)] = ap
        if not os.path.exists(ap):
            rep.add_warn(f"Missing PGA raster for RP={rp}: {ap}")
    rep.metrics["pga_rasters"] = pga_abs

    # Coastline
    coast = paths.get("coastline", {})
    coast_shp = coast.get("shp")
    coast_src = coast.get("source", "file")
    coast_abs = abspath(cfg, coast_shp) if coast_shp else None
    rep.metrics["coast_source"] = coast_src
    rep.metrics["coast_shp"] = coast_abs

    if coast_src == "file":
        if not coast_abs or not os.path.exists(coast_abs):
            rep.add_warn(f"Coastline source=file but shp missing: {coast_abs}")
        else:
            rep.add_ok(f"Found coastline shp: {coast_abs}")
    else:
        rep.add_ok(f"Coastline source is '{coast_src}' (not checking shp existence)")

    # Outputs
    transects_dir = abspath(cfg, outs["transects_dir"])
    summary_csv = abspath(cfg, outs["summary_csv"])
    maps_dir = abspath(cfg, outs.get("maps_dir", os.path.join(transects_dir, "maps")))
    rasters_dir = abspath(cfg, outs.get("rasters_dir", os.path.join(os.path.dirname(transects_dir), "rasters")))

    rep.metrics["transects_dir"] = transects_dir
    rep.metrics["summary_csv"] = summary_csv
    rep.metrics["maps_dir"] = maps_dir
    rep.metrics["rasters_dir"] = rasters_dir

    return {
        "bathy": bathy,
        "pga": pga_abs,
        "coast_shp": coast_abs,
        "transects_dir": transects_dir,
        "summary_csv": summary_csv,
        "maps_dir": maps_dir,
        "rasters_dir": rasters_dir,
    }


def check_crs_and_bounds(bathy_tif: str, coast_shp: Optional[str], rep: Report) -> None:
    with rasterio.open(bathy_tif) as src:
        bathy_crs = src.crs
        b = src.bounds
    rep.metrics["bathy_crs"] = str(bathy_crs)
    rep.metrics["bathy_bounds"] = (b.left, b.bottom, b.right, b.top)
    rep.add_ok(f"Bathy CRS={bathy_crs} bounds={b}")

    if coast_shp and os.path.exists(coast_shp):
        if gpd is None:
            rep.add_warn("geopandas not available; skipping coastline CRS/bounds checks.")
            return
        try:
            g = gpd.read_file(coast_shp)
            rep.metrics["coast_crs"] = str(g.crs)
            rep.metrics["coast_bounds"] = tuple(g.total_bounds.tolist())

            if g.crs is None:
                rep.add_warn("Coastline shapefile has no CRS defined.")
                return

            # Compare in bathy CRS
            if str(g.crs) != str(bathy_crs):
                rep.add_warn(
                    f"Coast CRS != Bathy CRS (will be reprojected in processing). coast={g.crs}, bathy={bathy_crs}"
                )
                g2 = g.to_crs(bathy_crs)
            else:
                g2 = g

            cb = g2.total_bounds  # minx,miny,maxx,maxy
            overlap = not (cb[2] < b.left or cb[0] > b.right or cb[3] < b.bottom or cb[1] > b.top)
            if overlap:
                rep.add_ok("Coastline bounds overlap bathy bounds (good).")
            else:
                rep.add_err("Coastline bounds do NOT overlap bathy bounds -> CRS/path likely wrong.")
        except Exception as e:
            rep.add_warn(f"Failed to read coastline shp: {e}")


def check_transects(
    transects_dir: str,
    bathy_tif: str,
    rep: Report,
    offshore_eps_m: float = 50.0,
    rmse_tol: float = 1e-2,
    offshore_tests_m=(200.0, 500.0, 1000.0, 2000.0, 3000.0),
    deepen_tol_m: float = 0.5,
) -> None:
    index_csv = os.path.join(transects_dir, "index.csv")
    rep.metrics["index_csv"] = index_csv

    if not os.path.exists(index_csv):
        rep.add_err(f"Missing index.csv: {index_csv}")
        return

    idx = pd.read_csv(index_csv)
    if "transect_id" not in idx.columns and "id" in idx.columns:
        idx = idx.rename(columns={"id": "transect_id"})

    need = {"transect_id", "x0", "y0", "nx", "ny", "npoints"}
    if not need.issubset(idx.columns):
        rep.add_err(f"index.csv missing columns {need}. Has {list(idx.columns)}")
        return

    with rasterio.open(bathy_tif) as src:
        nd = src.nodata

        def sample(xs, ys):
            v = np.array([a[0] for a in src.sample(list(zip(xs, ys)))], float)
            if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
                v[v == nd] = np.nan
            return v

        bad = []
        rmse_bad = []
        n_ok = 0

        for r in idx.itertuples(index=False):
            tid = int(r.transect_id)
            f = os.path.join(transects_dir, f"tr_{tid:04d}.csv")
            if not os.path.exists(f):
                bad.append((tid, "missing_transect_csv"))
                continue

            df = _safe_read_csv(f)
            if df is None or len(df) < 10:
                bad.append((tid, "empty_or_too_short"))
                continue

            for col in ("s_m", "x", "y", "depth"):
                if col not in df.columns:
                    bad.append((tid, f"missing_col:{col}"))
                    df = None
                    break
            if df is None:
                continue

            s = df["s_m"].to_numpy(float)
            z = df["depth"].to_numpy(float)

            if not np.all(np.diff(s) > 0):
                bad.append((tid, "s_not_strictly_increasing"))

            # offshore direction: robust to shelves/platforms
            def mean_depth(lo, hi):
                m = (s >= lo) & (s <= hi) & np.isfinite(z)
                return float(np.nanmean(z[m])) if np.any(m) else np.nan

            z0 = mean_depth(0, offshore_eps_m)
            smax = float(np.nanmax(s)) if np.any(np.isfinite(s)) else np.nan
            tests = [t for t in offshore_tests_m if np.isfinite(smax) and (t + offshore_eps_m) <= smax]

            z_tests = []
            for t in tests:
                zt = mean_depth(t - offshore_eps_m, t + offshore_eps_m)
                z_tests.append((t, zt))

            if np.isfinite(z0) and z_tests:
                ok_offshore = any(np.isfinite(zt) and (zt <= (z0 - float(deepen_tol_m))) for _, zt in z_tests)
                if not ok_offshore:
                    t_last, z_last = z_tests[-1]
                    bad.append(
                        (tid, f"not_offshore? z0={z0:.1f} z{int(t_last)}={z_last:.1f} (tol={deepen_tol_m}m)")
                    )

            # depth consistency (transect depth vs bathy sample)
            zs = sample(df["x"].to_numpy(float), df["y"].to_numpy(float))
            m = np.isfinite(z) & np.isfinite(zs)
            if m.sum() >= 20:
                rmse = float(np.sqrt(np.mean((z[m] - zs[m]) ** 2)))
                if rmse > rmse_tol:
                    rmse_bad.append((tid, rmse))
            else:
                bad.append((tid, "too_few_valid_points_for_rmse"))

            n_ok += 1

        rep.metrics["n_transects"] = int(len(idx))
        rep.metrics["transects_checked"] = int(n_ok)
        rep.metrics["transects_bad"] = bad[:]
        rep.metrics["transects_rmse_bad"] = rmse_bad[:]

        if len(bad) == 0:
            rep.add_ok("Transect CSV integrity + offshore direction checks: OK")
        else:
            rep.add_warn(f"Transect issues found: {len(bad)} (showing up to 12) -> {bad[:12]}")

        if len(rmse_bad) == 0:
            rep.add_ok(f"Depth column matches resampled bathy (RMSE <= {rmse_tol})")
        else:
            rep.add_warn(
                f"Depth mismatch vs bathy sampling: {len(rmse_bad)} transects RMSE>{rmse_tol} (up to 12): {rmse_bad[:12]}"
            )


def check_mc_outputs(transects_dir: str, rep: Report) -> None:
    patterns = {
        "trans_all": os.path.join(transects_dir, "tr_*_grilli_mcs.csv"),
        "rot_all": os.path.join(transects_dir, "tr_*_grilli_rot_mcs.csv"),
        "trans_reg": os.path.join(transects_dir, "tr_*_grilli_mcs_REGIONAL.csv"),
        "rot_reg": os.path.join(transects_dir, "tr_*_grilli_rot_mcs_REGIONAL.csv"),
    }

    out = {}
    for name, pat in patterns.items():
        files = sorted(glob.glob(pat))
        out[name] = {"n_files": len(files), "empty": [], "missing_cols": [], "kcrit_stats": None, "eta0_stats": None}
        if not files:
            continue

        k_all = []
        e_all = []

        for f in files:
            df = _safe_read_csv(f)
            if df is None:
                out[name]["empty"].append(os.path.basename(f))
                continue

            need = {"s0_m", "s1_m", "kcrit", "eta0_m"}
            if not need.issubset(df.columns):
                out[name]["missing_cols"].append((os.path.basename(f), sorted(list(need - set(df.columns)))))
                continue

            k = df["kcrit"].to_numpy(float)
            k = k[np.isfinite(k)]
            if k.size:
                k_all.append(k)

            e = df["eta0_m"].to_numpy(float)
            e = e[np.isfinite(e)]
            if e.size:
                e_all.append(e)

        if k_all:
            k = np.concatenate(k_all)
            out[name]["kcrit_stats"] = {
                "n": int(k.size),
                "p05": float(np.quantile(k, 0.05)),
                "p50": float(np.quantile(k, 0.50)),
                "p95": float(np.quantile(k, 0.95)),
                "frac_kcrit_le_0": float(np.mean(k <= 1e-12)),
                "min": float(k.min()),
                "max": float(k.max()),
            }
        if e_all:
            e = np.concatenate(e_all)
            out[name]["eta0_stats"] = {
                "n": int(e.size),
                "p50": float(np.quantile(e, 0.50)),
                "p95": float(np.quantile(e, 0.95)),
                "p99": float(np.quantile(e, 0.99)),
                "max": float(e.max()),
            }

        if out[name]["empty"]:
            rep.add_warn(f"{name}: empty/bad CSVs: {len(out[name]['empty'])} (ex: {out[name]['empty'][:5]})")
        if out[name]["missing_cols"]:
            rep.add_warn(
                f"{name}: missing required columns in {len(out[name]['missing_cols'])} files (ex: {out[name]['missing_cols'][:3]})"
            )

    # --- unit-consistency hint: compare *_all vs *_reg if both exist
    def _maybe_warn_units(kind: str):
        a = out.get(f"{kind}_all", {}).get("eta0_stats", None)
        b = out.get(f"{kind}_reg", {}).get("eta0_stats", None)
        if not (a and b):
            return
        p50_all = a.get("p50", np.nan)
        p50_reg = b.get("p50", np.nan)
        if not (np.isfinite(p50_all) and np.isfinite(p50_reg) and p50_reg > 0):
            return

        ratio = float(p50_all / p50_reg)
        rep.metrics[f"eta0_p50_ratio_{kind}_all_over_reg"] = ratio

        if 50.0 <= ratio <= 200.0:
            rep.add_warn(
                f"eta0 unit mismatch likely for '{kind}': median eta0(all)/eta0(reg) ~ {ratio:.1f}. "
                f"Looks like ALL may be in cm while REGIONAL is in m."
            )

        if np.isfinite(p50_all) and p50_all > 50.0:
            rep.add_warn(f"'{kind}_all' eta0 median is {p50_all:.2f} (very large) — check units / file type.")

    _maybe_warn_units("trans")
    _maybe_warn_units("rot")

    rep.metrics["mc_outputs"] = out
    rep.add_ok("MC outputs scan completed (see metrics.mc_outputs for details).")


def check_summary(summary_csv: str, rep: Report) -> None:
    if not os.path.exists(summary_csv):
        rep.add_warn(f"Summary CSV missing: {summary_csv}")
        return

    df = _safe_read_csv(summary_csv)
    if df is None:
        rep.add_err(f"Summary CSV is empty/bad: {summary_csv}")
        return

    if "transect_id" not in df.columns:
        rep.add_err("summary CSV missing 'transect_id'")
        return

    rps = _infer_rps_from_summary_columns(list(df.columns))
    rep.metrics["summary_rps"] = rps

    if not rps:
        rep.add_warn("Could not infer any RP from summary columns.")
        return

    # Basic range checks
    range_issues = 0
    for rp in rps:
        c = f"Pf_{rp}_any"
        if c in df.columns:
            v = df[c].to_numpy(float)
            m = np.isfinite(v)
            if np.any(v[m] < -1e-6) or np.any(v[m] > 1 + 1e-6):
                range_issues += 1
                rep.add_warn(f"{c} has values outside [0,1].")
    if range_issues == 0:
        rep.add_ok("Pf columns appear within [0,1].")

    # Monotonic Pf with RP (any)
    viol = 0
    for i in range(len(rps) - 1):
        a = df.get(f"Pf_{rps[i]}_any", pd.Series(np.nan, index=df.index)).to_numpy(float)
        b = df.get(f"Pf_{rps[i+1]}_any", pd.Series(np.nan, index=df.index)).to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        viol += int(np.sum(b[m] < a[m] - 1e-6))
    rep.metrics["Pf_monotonic_violations_any"] = int(viol)
    if viol == 0:
        rep.add_ok("Pf_any monotonic with RP (no violations).")
    else:
        rep.add_warn(
            f"Pf_any monotonic violations: {viol} (can happen with missing sampling, but large counts are suspicious)."
        )

    # Monotonic PGA mean with RP (any)
    viol_pga = 0
    pga_cols = [f"pga{rp}_g_center_mean_any" for rp in rps if f"pga{rp}_g_center_mean_any" in df.columns]
    for i in range(len(pga_cols) - 1):
        a = df[pga_cols[i]].to_numpy(float)
        b = df[pga_cols[i + 1]].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        viol_pga += int(np.sum(b[m] < a[m] - 1e-9))
    rep.metrics["PGA_monotonic_violations_any"] = int(viol_pga)
    if viol_pga == 0:
        rep.add_ok("PGAmean_any monotonic with RP (no violations).")
    else:
        rep.add_warn(f"PGAmean_any monotonic violations: {viol_pga}")

    # quick stats
    pf_stats = {}
    for rp in rps:
        c = f"Pf_{rp}_any"
        if c in df.columns:
            v = df[c].to_numpy(float)
            pf_stats[rp] = {"min": float(np.nanmin(v)), "p50": float(np.nanmedian(v)), "max": float(np.nanmax(v))}
    rep.metrics["summary_pf_any_stats"] = pf_stats

    # -------------------------
    # Hazard sanity checks (if present)
    # -------------------------
    cols = list(df.columns)

    # helper: "050cm" -> 0.5
    def _tag_cm_to_m(tag: str) -> float:
        try:
            return float(int(tag.replace("cm", ""))) / 100.0
        except Exception:
            return np.nan

    # (i) PfH ordering vs Pf and PfT (when available)
    pfh_re = re.compile(r"^PfH(\d+cm)_(\d+)_([a-z]+)$")  # PfH050cm_100_any
    pfh_defs = []  # (tag, rp, typ, col)
    tags = set()

    for c in cols:
        m = pfh_re.match(c)
        if m:
            tag, rp, typ = m.group(1), m.group(2), m.group(3)
            tags.add(tag)
            pfh_defs.append((tag, rp, typ, c))

    if pfh_defs:
        viol_order = 0

        # eta_min is stored in summary (your step 5 writes eta_min_m)
        eta_min_arr = df["eta_min_m"].to_numpy(float) if "eta_min_m" in df.columns else None

        for tag, rp, typ, c_pfh in pfh_defs:
            c_pf = f"Pf_{rp}_{typ}"
            c_pft = f"PfT_{rp}_{typ}"
            if (c_pf not in df.columns) or (c_pft not in df.columns):
                continue

            pf = df[c_pf].to_numpy(float)
            pft = df[c_pft].to_numpy(float)
            pfh = df[c_pfh].to_numpy(float)

            # always enforce PfT <= Pf
            m0 = np.isfinite(pf) & np.isfinite(pft)
            viol_order += int(np.sum(pft[m0] > pf[m0] + 1e-12))

            # only enforce PfH <= PfT if H >= eta_min (otherwise PfH can be larger)
            Hm = _tag_cm_to_m(tag)
            if np.isfinite(Hm):
                if eta_min_arr is not None:
                    m1 = np.isfinite(pfh) & np.isfinite(pft) & np.isfinite(eta_min_arr) & (Hm >= (eta_min_arr - 1e-12))
                else:
                    m1 = np.isfinite(pfh) & np.isfinite(pft)
                viol_order += int(np.sum(pfh[m1] > pft[m1] + 1e-12))

        rep.metrics["PfH_PfT_Pf_order_violations"] = int(viol_order)
        if viol_order == 0:
            rep.add_ok("Hazard ordering OK where available: PfT <= Pf (and PfH <= PfT when H >= eta_min).")
        else:
            rep.add_warn(f"Hazard ordering violations: {viol_order}")

    # (ii) RPH ≈ 1/AERH where both exist
    aerr = 0.0
    ncheck = 0
    for c in cols:
        if c.startswith("AERH") and ("_" in c):
            c_r = "RPH" + c[len("AERH") :]
            if c_r in df.columns:
                a = df[c].to_numpy(float)
                r = df[c_r].to_numpy(float)
                m = np.isfinite(a) & (a > 0) & np.isfinite(r)
                if np.any(m):
                    err = np.nanmax(np.abs(r[m] - 1.0 / a[m]))
                    aerr = max(aerr, float(err))
                    ncheck += 1

    rep.metrics["RPH_reciprocal_max_abs_err"] = float(aerr) if ncheck else np.nan
    if ncheck:
        if aerr < 1e-6:
            rep.add_ok(f"RPH reciprocal check OK (max |RPH-1/AERH|={aerr:.2e})")
        else:
            rep.add_warn(f"RPH reciprocal check: max |RPH-1/AERH|={aerr:.2e} (should be tiny)")

    # (iii) If multiple thresholds exist: PfH decreases with higher threshold
    if len(tags) >= 2 and pfh_defs:
        tag_m = sorted([(t, _tag_cm_to_m(t)) for t in tags], key=lambda x: x[1])
        tag_m = [(t, m) for (t, m) in tag_m if np.isfinite(m)]
        if len(tag_m) >= 2:
            group = {}
            for tag, rp, typ, col in pfh_defs:
                group.setdefault((rp, typ), {})[tag] = col

            viol_thr = 0
            for (rp, typ), dct in group.items():
                for i in range(len(tag_m) - 1):
                    t1, _ = tag_m[i]
                    t2, _ = tag_m[i + 1]
                    c1 = dct.get(t1, None)
                    c2 = dct.get(t2, None)
                    if (c1 is None) or (c2 is None):
                        continue
                    v1 = df[c1].to_numpy(float)
                    v2 = df[c2].to_numpy(float)
                    m = np.isfinite(v1) & np.isfinite(v2)
                    # higher threshold (t2) should not exceed lower (t1)
                    viol_thr += int(np.sum(v2[m] > v1[m] + 1e-12))

            rep.metrics["PfH_threshold_monotonic_violations"] = int(viol_thr)
            if viol_thr == 0:
                rep.add_ok("PfH threshold monotonicity OK (higher H => lower/equal PfH)")
            else:
                rep.add_warn(f"PfH threshold monotonicity violations: {viol_thr}")


def make_plots(cfg: Dict[str, Any], resolved: Dict[str, Any], rep: Report) -> None:
    if plt is None:
        rep.add_warn("matplotlib not available; skipping plots.")
        return

    out_dir = abspath(cfg, cfg["outputs"].get("maps_dir", os.path.join(resolved["transects_dir"], "maps")))
    sanity_dir = os.path.join(out_dir, "_sanity")
    os.makedirs(sanity_dir, exist_ok=True)

    summ = resolved["summary_csv"]
    df = _safe_read_csv(summ)
    if df is None:
        rep.add_warn("No summary to plot.")
        return

    rps = _infer_rps_from_summary_columns(list(df.columns))
    if not rps:
        rep.add_warn("No RPs inferred; skipping summary plots.")
        return

    rp_max = rps[-1]
    pf_col = f"Pf_{rp_max}_any"
    if pf_col in df.columns:
        v = df[pf_col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            plt.figure()
            plt.hist(v, bins=30)
            plt.xlabel(pf_col)
            plt.ylabel("count")
            plt.title(f"Histogram {pf_col}")
            p = os.path.join(sanity_dir, f"hist_{pf_col}.png")
            plt.tight_layout()
            plt.savefig(p, dpi=200)
            plt.close()
            rep.add_ok(f"Wrote plot: {p}")

    eta_col = f"eta0_fail{rp_max}_p99_m_any"
    if eta_col not in df.columns:
        for c in df.columns:
            if c.startswith("eta0_fail") and c.endswith("_any"):
                eta_col = c
                break
    if eta_col in df.columns:
        v = df[eta_col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            plt.figure()
            plt.hist(v, bins=30)
            plt.xlabel(eta_col)
            plt.ylabel("count")
            plt.title(f"Histogram {eta_col}")
            p = os.path.join(sanity_dir, f"hist_{eta_col}.png")
            plt.tight_layout()
            plt.savefig(p, dpi=200)
            plt.close()
            rep.add_ok(f"Wrote plot: {p}")

    pga_col = f"pga{rp_max}_g_center_mean_any"
    if pf_col in df.columns and pga_col in df.columns:
        x = df[pga_col].to_numpy(float)
        y = df[pf_col].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 5:
            plt.figure()
            plt.scatter(x[m], y[m], s=30)
            plt.xlabel(pga_col)
            plt.ylabel(pf_col)
            plt.title(f"{pf_col} vs {pga_col}")
            p = os.path.join(sanity_dir, f"scatter_{pf_col}_vs_{pga_col}.png")
            plt.tight_layout()
            plt.savefig(p, dpi=200)
            plt.close()
            rep.add_ok(f"Wrote plot: {p}")

    rep.metrics["sanity_plots_dir"] = sanity_dir


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 7: run sanity checks for the whole pipeline (config-driven).")
    ap.add_argument("--config", required=True, help="Path to cases/config.json")
    ap.add_argument("--plots", action="store_true", help="Write a few quick sanity plots into <maps_dir>/_sanity")
    ap.add_argument("--fail-on-warn", action="store_true", help="Exit non-zero if any warnings are present")
    ap.add_argument("--rmse-tol", type=float, default=1e-2, help="Depth-vs-bathy RMSE tolerance (meters)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rep = Report()

    resolved = check_inputs(cfg, rep)
    if resolved:
        check_crs_and_bounds(resolved["bathy"], resolved.get("coast_shp", None), rep)

        if os.path.exists(resolved["transects_dir"]) and os.path.exists(resolved["bathy"]):
            check_transects(resolved["transects_dir"], resolved["bathy"], rep, rmse_tol=float(args.rmse_tol))
        else:
            rep.add_warn("Skipping transect checks (missing transects_dir or bathy).")

        if os.path.exists(resolved["transects_dir"]):
            check_mc_outputs(resolved["transects_dir"], rep)

        check_summary(resolved["summary_csv"], rep)

        if args.plots:
            make_plots(cfg, resolved, rep)

    # Print report
    print("\n==================== STEP 7 SANITY REPORT ====================")
    print(rep.summary())

    if rep.err:
        print("\n[ERRORS]")
        for s in rep.err[:50]:
            print(" -", s)
        if len(rep.err) > 50:
            print(" ...", len(rep.err) - 50, "more")

    if rep.warn:
        print("\n[WARNINGS]")
        for s in rep.warn[:50]:
            print(" -", s)
        if len(rep.warn) > 50:
            print(" ...", len(rep.warn) - 50, "more")

    if rep.ok:
        print("\n[OK]")
        for s in rep.ok[:50]:
            print(" -", s)

    # Write JSON report next to summary CSV if possible
    out_json = None
    try:
        summ = rep.metrics.get("summary_csv", None)
        if summ:
            out_json = os.path.join(os.path.dirname(summ), "sanity_report.json")
        else:
            out_json = os.path.join(os.getcwd(), "sanity_report.json")
        with open(out_json, "w") as f:
            json.dump({"ok": rep.ok, "warn": rep.warn, "err": rep.err, "metrics": rep.metrics}, f, indent=2)
        print("\n[OK] wrote JSON report:", out_json)
    except Exception as e:
        print("\n[WARN] could not write JSON report:", e)

    # Exit code
    if rep.err:
        raise SystemExit(2)
    if args.fail_on_warn and rep.warn:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()