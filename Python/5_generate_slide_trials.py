#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

IN_DIR = "transects"
CAND = os.path.join(IN_DIR, "candidate_segments.csv")
OUT  = os.path.join(IN_DIR, "slide_trials.csv")

# -------------------------
# How many trials?
# -------------------------
N_TRIALS_PER_SEG = 300   # start with 100–300; later 1000–5000+

# -------------------------
# "Present-day regional" geometry priors (edit later)
# -------------------------
# Length (along-slope, along transect)
L_MIN = 300.0
L_MAX = 4000.0          # cap for regional (not mega)

# Thickness-to-length ratio (very common simple prior)
# You can tighten/adjust later
TL_MIN, TL_MAX = 0.01, 0.05     # T = TL * L (translational-ish)
# Width-to-length ratio
WL_MIN, WL_MAX = 0.5, 2.0       # W = WL * L

# Optional: reject very shallow/deep head positions (depth at s0)
D0_MIN = -1800.0
D0_MAX = -80.0

SEED = 123
rng = np.random.default_rng(SEED)

def volume_proxy(L, W, T):
    # crude proxy (fine for now; not a final volume model)
    return 0.5 * L * W * T

cand = pd.read_csv(CAND)
if len(cand) == 0:
    raise RuntimeError("candidate_segments.csv is empty")

# cache transects
tr_cache = {}

rows = []
trial_id = 0

for seg_id, r in enumerate(cand.itertuples(index=False), start=1):
    tr_path = os.path.join(IN_DIR, r.csv)
    if r.csv not in tr_cache:
        tr_cache[r.csv] = pd.read_csv(tr_path)
    tr = tr_cache[r.csv]

    sA = float(r.s_start_m)
    sB = float(r.s_end_m)
    seg_len = float(r.len_m)

    # maximum feasible L in this segment
    Lmax_here = min(L_MAX, seg_len)
    if Lmax_here < L_MIN:
        continue

    # pre-subset the segment
    mask_seg = (tr["s_m"] >= sA) & (tr["s_m"] <= sB)
    tr_seg = tr.loc[mask_seg].copy()
    if len(tr_seg) < 5:
        continue

    s_m = tr["s_m"].to_numpy(float)  # full arrays (for slicing by s0/s1)
    # Generate trials
    made = 0
    attempts = 0
    max_attempts = N_TRIALS_PER_SEG * 30

    while made < N_TRIALS_PER_SEG and attempts < max_attempts:
        attempts += 1

        # sample slide length
        L = rng.uniform(L_MIN, Lmax_here)

        # choose window inside the candidate segment
        s0 = rng.uniform(sA, sB - L)
        s1 = s0 + L

        m = (tr["s_m"] >= s0) & (tr["s_m"] <= s1)
        patch = tr.loc[m]
        if len(patch) < 5:
            continue

        # depth at head (closest point to s0)
        i0 = np.argmin(np.abs(patch["s_m"].to_numpy(float) - s0))
        d0 = float(patch["depth"].iloc[i0])
        if not (D0_MIN <= d0 <= D0_MAX):
            continue

        # thickness and width
        TL = rng.uniform(TL_MIN, TL_MAX)
        T  = TL * L
        WL = rng.uniform(WL_MIN, WL_MAX)
        W  = WL * L

        # descriptors
        depth_mean = float(patch["depth"].mean())
        slope_mean = float(patch["slope_deg"].mean())
        slope_max  = float(patch["slope_deg"].max())
        rough_mean = float(patch["rough"].mean())
        frac_sed   = float((patch["cls"] == 0).mean())
        xc = float(patch["x"].mean())
        yc = float(patch["y"].mean())

        rows.append({
            "trial_id": trial_id,
            "transect_id": int(r.transect_id),
            "segment_row": int(seg_id),
            "src_csv": str(r.csv),
            "s0_m": float(s0),
            "s1_m": float(s1),
            "L_m": float(L),
            "W_m": float(W),
            "T_m": float(T),
            "depth0_m": float(d0),
            "depth_mean_m": float(depth_mean),
            "slope_mean_deg": float(slope_mean),
            "slope_max_deg": float(slope_max),
            "rough_mean": float(rough_mean),
            "frac_sed": float(frac_sed),
            "x_center": xc,
            "y_center": yc,
            "volume_proxy_m3": float(volume_proxy(L, W, T)),
        })
        trial_id += 1
        made += 1

print(f"[INFO] Generated {len(rows)} trials (target {N_TRIALS_PER_SEG} per segment; segments={len(cand)})")

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"[INFO] Saved -> {OUT}")

if len(out):
    print("\n=== QC ===")
    print(out[["L_m","W_m","T_m","depth0_m","slope_mean_deg","volume_proxy_m3"]].describe())
    print("\nTrials per transect (top 10):")
    print(out.groupby("transect_id").size().sort_values(ascending=False).head(10))
