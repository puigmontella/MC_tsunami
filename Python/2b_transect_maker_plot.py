#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# EDIT THESE
# -------------------------
TRANSECTS_DIR = "transects"
TRANSECT_IDS = [0,13,7,11,12,9,8,27,15]   # <-- put your list here
OUT_DIR = os.path.join(TRANSECTS_DIR, "profiles_png")
INVERT_DEPTH = True          # True => 0 at top, deeper down
# -------------------------

os.makedirs(OUT_DIR, exist_ok=True)

for tid in TRANSECT_IDS:
    f = os.path.join(TRANSECTS_DIR, f"tr_{tid:04d}.csv")
    if not os.path.exists(f):
        print(f"[SKIP] missing: {f}")
        continue

    df = pd.read_csv(f)

    s = df["s_m"].to_numpy(float)
    depth = df["depth"].to_numpy(float)       # negative
    slope = df["slope_deg"].to_numpy(float)
    rough = df["rough"].to_numpy(float)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # depth
    ax[0].plot(s, depth, linewidth=1.5)
    ax[0].set_ylabel("depth (m)")
    if INVERT_DEPTH:
        ax[0].invert_yaxis()
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f"Transect {tid:04d}")

    # slope
    ax[1].plot(s, slope, linewidth=1.5)
    ax[1].set_ylabel("slope (deg)")
    ax[1].grid(True, alpha=0.3)

    # roughness
    ax[2].plot(s, rough, linewidth=1.5)
    ax[2].set_ylabel("roughness")
    ax[2].set_xlabel("s along transect (m)")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, f"profile_tr_{tid:04d}.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[OK] {out_png}")

print("[DONE]")
