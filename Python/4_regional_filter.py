#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np

inp = sys.argv[1]  # e.g. transects/tr_0003_grilli_mcs.csv
df = pd.read_csv(inp)

# --- regional filters (edit) ---
L_MAX = 8000.0
T_MAX = 150.0
END_MARGIN = 200.0  # meters (avoid clipped windows)
# -------------------------------

smax = df["s1_m"].max()
df_reg = df[(df["L_m"] <= L_MAX) &
            (df["T_m"] <= T_MAX) &
            (df["s1_m"] <= (smax - END_MARGIN))].copy()

out = inp.replace(".csv", "_REGIONAL.csv")
df_reg.to_csv(out, index=False)

print(f"[OK] regional subset: {len(df_reg)} / {len(df)}  -> {out}")

def summ(x, name):
    print(f"\n=== {name} ===")
    print(x.describe(percentiles=[0.05,0.5,0.9,0.95,0.99]))

summ(df["kcrit"], "kcrit (ALL)")
summ(df_reg["kcrit"], "kcrit (REGIONAL)")

summ(df["eta0_m"], "eta0_m (ALL)")
summ(df_reg["eta0_m"], "eta0_m (REGIONAL)")







