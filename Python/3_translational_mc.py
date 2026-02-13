#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Grilli-style MCS (TRANSLATIONAL)
# ----------------------------
# Uses:
#  - Infinite-slope pseudo-static FS (Eq. 7)
#  - Ru sampling (Eqs. 13–15)
#  - Geometry ranges (Table 1 ranges) + lognormal fits (Figs. 11–13) as defaults
#
# Output:
#  - A CSV of accepted trials, including k_crit (critical pseudo-static coefficient)
#  - Some summary stats

RHO_W = 1025.0  # kg/m3 (sea water)

def lognormal_mu_sigma(mean, std):
    """Convert mean/std in linear space to mu/sigma of underlying lognormal."""
    mean = float(mean); std = float(std)
    var = std**2
    sigma2 = np.log(1.0 + var/(mean**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5*sigma2
    return mu, sigma

def sample_lognormal_trunc(rng, mean, std, vmin, vmax, max_tries=10000):
    """Scalar truncated lognormal by rejection sampling."""
    mu, sig = lognormal_mu_sigma(mean, std)
    for _ in range(max_tries):
        x = rng.lognormal(mu, sig)
        if (x >= vmin) and (x <= vmax):
            return float(x)
    return None

def compute_beta_from_segment(s_seg, d_seg):
    """
    Robust slope angle beta from linear fit of depth (positive) vs distance.
    beta = atan(dd/ds)
    """
    if len(s_seg) < 3:
        return None
    # least-squares slope
    s0 = s_seg - s_seg.mean()
    d0 = d_seg - d_seg.mean()
    denom = np.dot(s0, s0)
    if denom <= 0:
        return None
    m = np.dot(s0, d0) / denom  # dd/ds
    beta = np.arctan(m)         # radians
    return float(beta)

def pick_start_index_for_depth(rng, s, depth_pos, d_target, L_target, tol_list=(20, 50, 100, 200, 300)):
    """
    Pick a start index i such that depth_pos[i] ~ d_target within tolerance,
    and enough room remains to fit L_target.
    """
    smax = s[-1]
    for tol in tol_list:
        candidates = np.where(
            (np.abs(depth_pos - d_target) <= tol) &
            (s <= smax - L_target)
        )[0]
        if len(candidates) > 0:
            return int(rng.choice(candidates))
    return None

def ru_translational_grilli(rng, beta_deg, phi_deg, eps=0.5):
    """
    Grilli translational Ru (Eqs. 13–15):
      Rut = rtV + alpha_t (rt - rtV), alpha_t = R + eps, -1<=alpha_t<=1
      rtV = 0.9 - 0.03 beta
      rt  = 0.991 - 0.033 beta + 0.0003/phi
    R ~ N(0, 0.25)
    """
    R = rng.normal(0.0, 0.25)
    alpha_t = np.clip(R + eps, -1.0, 1.0)

    rtV = 0.9   - 0.03  * beta_deg
    rt  = 0.991 - 0.033 * beta_deg + 0.0003 / max(phi_deg, 1e-6)

    Rut = rtV + alpha_t * (rt - rtV)

    # Physical bounds used in paper discussion (and to avoid nonsense)
    Rut = np.clip(Rut, 0.0, min(float(rt), 0.99))
    return float(Rut)

def fs_translational(beta, phi, s_spec, Ru, k):
    """
    Eq. (7):
    FS = ((s-1)(1-Ru) - k s tan(beta)) / ((s-1) tan(beta) + k s tan(phi))
    """
    tb = np.tan(beta)
    tp = np.tan(phi)
    num = (s_spec - 1.0) * (1.0 - Ru) - k * s_spec * tb
    den = (s_spec - 1.0) * tb + k * s_spec * tp
    if den == 0:
        return np.nan
    return float(num / den)

def kcrit_from_fs1(beta, phi, s_spec, Ru):
    """
    Solve Eq. (7) for k when FS=1:
      (s-1)(1-Ru) - k s tanb = (s-1)tanb + k s tanp
      k = (s-1)*((1-Ru) - tanb) / ( s*(tanb + tanp) )
    """
    tb = np.tan(beta)
    tp = np.tan(phi)
    denom = s_spec * (tb + tp)
    if denom <= 0:
        return np.nan
    k = (s_spec - 1.0) * ((1.0 - Ru) - tb) / denom
    return float(k)

def width_translational_grilli(rng, L):
    """
    Grilli width rule (Eqs. 16–17 text):
    translational width ~ 25% of length with ±20% variability.
    Implemented as: w = (0.2 + 0.1*R)*L, R~U[0,1]
    (This matches 0.25 ± 20% => [0.2, 0.3].)
    """
    R = rng.random()
    return float((0.2 + 0.1 * R) * L)

def eta0_translational_grilli(beta, T, L, d_center, s_spec, w):
    """
    Eq. (18) + (20) + (22):
      SoT = (pi/2) * L * (s+1)
      lambda0T = sqrt( (pi*L*d*(s+1)^2)/(2*sin(beta)) * ((s-1)/s) )
      eta0T = SoT*(0.0574 - 0.0431*sin(beta))*(T/L)*(L*sin(beta)/d)^1.25
              * (1-exp(-2.2*(s-1))) * w/(w + lambda0T)
    """
    sb = np.sin(beta)
    if sb <= 0 or d_center <= 0 or L <= 0:
        return np.nan

    SoT = 0.5 * np.pi * L * (s_spec + 1.0)
    lambda0T = np.sqrt(
        (np.pi * L * d_center * (s_spec + 1.0)**2) / (2.0 * sb) * ((s_spec - 1.0) / s_spec)
    )

    coef = (0.0574 - 0.0431 * sb)
    term1 = (T / L)
    term2 = (L * sb / d_center) ** 1.25
    term3 = (1.0 - np.exp(-2.2 * (s_spec - 1.0)))
    term4 = w / (w + lambda0T)

    eta0 = SoT * coef * term1 * term2 * term3 * term4
    return float(eta0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transect_csv", help="e.g. transects/tr_0014.csv")
    ap.add_argument("--n", type=int, default=200000, help="number of MCS trials")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output csv path (default: alongside input)")
    ap.add_argument("--keep_fs_checks", action="store_true",
                    help="Keep only trials passing FS1>1 and FS2>1 checks (recommended)")
    # Geometry distributions: default from Grilli validation fits (Figs. 12-13)
    ap.add_argument("--d_mean", type=float, default=1247.0, help="depth mean (m)")
    ap.add_argument("--d_std",  type=float, default=485.0,  help="depth std (m)")
    ap.add_argument("--L_mean", type=float, default=5749.0, help="length mean (m)")
    ap.add_argument("--L_std",  type=float, default=3024.0, help="length std (m)")
    ap.add_argument("--d_min", type=float, default=120.0)
    ap.add_argument("--d_max", type=float, default=2500.0)
    ap.add_argument("--L_min", type=float, default=1000.0)
    ap.add_argument("--L_max", type=float, default=50000.0)
    # Thickness ratio range (Table 1 for translational)
    ap.add_argument("--T_over_L_min", type=float, default=0.01)
    ap.add_argument("--T_over_L_max", type=float, default=0.04)
    # Sediment density (Fig. 11 fit)
    ap.add_argument("--rho_mean", type=float, default=1816.6)
    ap.add_argument("--rho_std",  type=float, default=39.9)
    # Phi' prior (edit for Tenerife if you later have constraints)
    ap.add_argument("--phi_mean", type=float, default=31.7)
    ap.add_argument("--phi_std",  type=float, default=2.0)
    ap.add_argument("--phi_min",  type=float, default=28.0)
    ap.add_argument("--phi_max",  type=float, default=40.0)
    args = ap.parse_args()

    tr_path = Path(args.transect_csv)
    df = pd.read_csv(tr_path)

    if not {"s_m", "depth"}.issubset(df.columns):
        raise ValueError("Transect CSV must have at least columns: s_m, depth")

    s = df["s_m"].to_numpy(float)
    depth = df["depth"].to_numpy(float)        # negative in your files
    depth_pos = -depth                         # positive depth below sea level (m)

    # basic cleaning
    m = np.isfinite(s) & np.isfinite(depth_pos)
    s = s[m]; depth_pos = depth_pos[m]
    if len(s) < 10:
        raise ValueError("Transect too short after cleaning.")

    rng = np.random.default_rng(args.seed)

    out_rows = []
    n_attempt = 0
    n_accept = 0

    for it in range(args.n):
        n_attempt += 1

        # sample depth & length (truncated lognormal)
        d0 = sample_lognormal_trunc(rng, args.d_mean, args.d_std, args.d_min, args.d_max)
        L  = sample_lognormal_trunc(rng, args.L_mean, args.L_std, args.L_min, args.L_max)
        if (d0 is None) or (L is None):
            continue

        i0 = pick_start_index_for_depth(rng, s, depth_pos, d0, L)
        if i0 is None:
            continue

        s_end = s[i0] + L
        i1 = int(np.searchsorted(s, s_end, side="left"))
        if i1 >= len(s):
            continue
        if i1 <= i0 + 2:
            continue

        # actual realized length
        L_eff = float(s[i1] - s[i0])

        # slope angle beta from segment
        beta = compute_beta_from_segment(s[i0:i1+1], depth_pos[i0:i1+1])
        if beta is None or not np.isfinite(beta):
            continue
        beta_deg = np.degrees(beta)

        # sample phi', rho_s
        phi_deg = float(np.clip(rng.normal(args.phi_mean, args.phi_std), args.phi_min, args.phi_max))
        phi = np.radians(phi_deg)

        rho_s = float(rng.normal(args.rho_mean, args.rho_std))
        rho_s = float(np.clip(rho_s, 1600.0, 2200.0))  # just sanity bounds
        s_spec = rho_s / RHO_W

        # thickness
        T_over_L = float(rng.uniform(args.T_over_L_min, args.T_over_L_max))
        T = T_over_L * L_eff

        # Ru (Grilli translational)
        Ru = ru_translational_grilli(rng, beta_deg, phi_deg, eps=0.5)

        # Stage checks (Grilli validation logic)
        FS1 = np.tan(phi) / np.tan(beta) if np.tan(beta) > 0 else np.nan
        FS2 = (1.0 - Ru) * FS1

        if args.keep_fs_checks:
            if not (np.isfinite(FS1) and np.isfinite(FS2)):
                continue
            if (FS1 <= 1.0) or (FS2 <= 1.0):
                continue

        # kcrit from FS=1
        kcrit = kcrit_from_fs1(beta, phi, s_spec, Ru)
        if not np.isfinite(kcrit):
            continue

        # Depth at slide "center" (for eta0), use midpoint index
        ic = (i0 + i1) // 2
        d_center = float(depth_pos[ic])

        # width (Grilli rule for translational)
        w = width_translational_grilli(rng, L_eff)

        # optional eta0 (Grilli)
        eta0 = eta0_translational_grilli(beta, T, L_eff, d_center, s_spec, w)

        out_rows.append({
            "trial": it,
            "i0": i0,
            "i1": i1,
            "s0_m": float(s[i0]),
            "s1_m": float(s[i1]),
            "L_m": L_eff,
            "d0_m_target": d0,
            "d0_m_real": float(depth_pos[i0]),
            "d_center_m": d_center,
            "beta_deg": beta_deg,
            "phi_deg": phi_deg,
            "rho_s": rho_s,
            "s_spec": s_spec,
            "T_m": T,
            "T_over_L": T_over_L,
            "Ru": Ru,
            "FS1": float(FS1),
            "FS2": float(FS2),
            "kcrit": float(kcrit),
            "w_m": w,
            "eta0_m": float(eta0),
        })
        n_accept += 1

    out = pd.DataFrame(out_rows)

    if args.out is None:
        out_path = tr_path.with_name(tr_path.stem + "_grilli_mcs.csv")
    else:
        out_path = Path(args.out)

    out.to_csv(out_path, index=False)

    print(f"[DONE] Attempts: {n_attempt}  Accepted: {n_accept}")
    print(f"[OUT] {out_path}")
    if len(out) > 0:
        print("\n=== kcrit summary ===")
        print(out["kcrit"].describe(percentiles=[0.05, 0.1, 0.5, 0.9, 0.95]))
        print("\n=== eta0 summary (m) ===")
        print(out["eta0_m"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

if __name__ == "__main__":
    main()
