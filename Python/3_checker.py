#!/usr/bin/env python3
import numpy as np
import pandas as pd

CSV = "transects/tr_0003_grilli_mcs.csv"   # <-- change if needed

df = pd.read_csv(CSV)
print("Columns:", df.columns.tolist())

def pick(*names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of these columns found: {names}")

col_beta = pick("beta_deg")
col_phi  = pick("phi_deg")
col_ru   = pick("Ru")
col_s    = pick("s_spec")
col_kc   = pick("kcrit")

def FS(beta_deg, phi_deg, s, Ru, k):
    beta = np.deg2rad(beta_deg)
    phi  = np.deg2rad(phi_deg)
    num = ((s - 1.0) * (1.0 - Ru) - k * s * np.tan(beta)) * np.tan(phi)
    den = (s - 1.0) * np.tan(beta) + k * s
    return num / den

sub = df.sample(n=min(30, len(df)), random_state=0).copy()

eps = 0.002
fs0    = FS(sub[col_beta], sub[col_phi], sub[col_s], sub[col_ru], 0.0)
fsk    = FS(sub[col_beta], sub[col_phi], sub[col_s], sub[col_ru], sub[col_kc])
fsk_lo = FS(sub[col_beta], sub[col_phi], sub[col_s], sub[col_ru], np.maximum(sub[col_kc] - eps, 0.0))
fsk_hi = FS(sub[col_beta], sub[col_phi], sub[col_s], sub[col_ru], sub[col_kc] + eps)

out = pd.DataFrame({
    "kcrit": sub[col_kc].to_numpy(),
    "FS(k=0)": fs0,
    "FS(kcrit)": fsk,
    "FS(kcrit-eps)": fsk_lo,
    "FS(kcrit+eps)": fsk_hi,
    "Ru": sub[col_ru].to_numpy(),
    "beta_deg": sub[col_beta].to_numpy(),
    "phi_deg": sub[col_phi].to_numpy(),
    "s_spec": sub[col_s].to_numpy(),
})

print("\n=== FS consistency check (should have FS(kcrit) ~ 1) ===")
print(out.describe())

worst = (out["FS(kcrit)"] - 1.0).abs().sort_values(ascending=False).head(5).index
print("\nWorst |FS(kcrit)-1| cases:")
print(out.loc[worst].to_string(index=False))



df = pd.read_csv("transects/tr_0003_grilli_mcs.csv")
top = df.sort_values("eta0_m", ascending=False).head(20)
print(top[["eta0_m","kcrit","L_m","T_over_L","T_m","d_center_m","beta_deg","Ru","phi_deg","w_m","s0_m","s1_m"]].to_string(index=False))
