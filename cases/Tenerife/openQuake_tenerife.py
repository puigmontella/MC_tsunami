#!/usr/bin/env python3
import numpy as np
import pandas as pd
from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD

# -------------------
# Parameters (edit here)
# -------------------
n_magnitudes = 200000      # number of magnitudes to generate
mmin = 4.5                  # minimum magnitude
mmax = 6.4                  # maximum magnitude
b = 0.8                     # GR b-value (controls tail: larger b -> fewer big events)
dm = 0.1                    # magnitude bin width
seed = 1234
jitter = True               # add uniform jitter in [-0.5*dm, +0.5*dm]
out_csv = "inputs/parameters/magnitude.csv"
# -------------------

rng = np.random.default_rng(seed)

def sample_mag_gr(rng, mmin, mmax, b, dm, jitter=True):
    """
    Sample Mw from a truncated Gutenberg–Richter distribution using OpenQuake hazardlib.

    - Discretize Mw in bins of width dm (centers mmin, mmin+dm, ..., mmax)
    - Assign GR weights ~ 10^(-b*M) (up to normalization)
    - Sample a bin, optionally jitter inside that bin
    """
    mfd = TruncatedGRMFD(
        min_mag=float(mmin),
        max_mag=float(mmax),
        bin_width=float(dm),
        a_val=0.0,       # absolute rate irrelevant for conditional sampling
        b_val=float(b),
    )

    mags_rates = list(mfd.get_annual_occurrence_rates())  # [(Mw_center, rate), ...]
    mags  = np.array([mr[0] for mr in mags_rates], dtype=float)
    rates = np.array([mr[1] for mr in mags_rates], dtype=float)

    p = rates / rates.sum()               # convert rates -> probabilities
    M = float(rng.choice(mags, p=p))      # pick one magnitude bin center

    if jitter and dm > 0:
        M += float(rng.uniform(-0.5 * dm, 0.5 * dm))  # jitter within bin
        M = float(np.clip(M, mmin, mmax))             # keep inside bounds

    return M

# Generate the pool
M = np.array(
    [sample_mag_gr(rng, mmin, mmax, b, dm, jitter=jitter) for _ in range(n_magnitudes)],
    dtype=float
)

# Save
df = pd.DataFrame({
    "M_evt": M,
    "mmin": mmin,
    "mmax": mmax,
    "b": b,
    "dm": dm,
    "seed": seed,
})
df.to_csv(out_csv, index=False)

print(f"[OK] wrote {out_csv}")
print("[INFO] min / mean / max:", float(M.min()), float(M.mean()), float(M.max()))
