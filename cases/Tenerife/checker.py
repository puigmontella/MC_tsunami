import pandas as pd, numpy as np

fa = "outputs/coast/coast_runup_eqRP100_trans_areas.csv"
ft = "outputs/coast/coast_runup_eqRP100_trans_transects.csv"

A = pd.read_csv(fa)
T = pd.read_csv(ft)

# merge on coast_id (and x,y if you like)
M = A.merge(T, on="coast_id", suffixes=("_areas","_tran"))

# pick the metric to compare
col = "RunUp_2475yr"
for k in [col, "RunUp_p99", "RunUp_p90", "RunUp_p50", "lambda_RunUp_ge_thr_1yr"]:
    if (k+"_areas") in M.columns and (k+"_tran") in M.columns:
        da = pd.to_numeric(M[k+"_areas"], errors="coerce")
        dt = pd.to_numeric(M[k+"_tran"], errors="coerce")
        d  = da - dt
        good = np.isfinite(d)

        print("\n==", k, "==")
        print("N valid:", int(good.sum()), " / ", len(M))
        if good.any():
            print("delta median:", float(np.nanmedian(d)))
            print("delta p10/p90:", float(np.nanquantile(d,0.1)), float(np.nanquantile(d,0.9)))
            print("delta max abs:", float(np.nanmax(np.abs(d))))
            j = int(np.nanargmax(np.abs(d.to_numpy())))
            print("worst coast_id:", int(M.loc[j,"coast_id"]),
                  "areas:", float(da.iloc[j]), "tran:", float(dt.iloc[j]), "delta:", float(d.iloc[j]))

# also show RP_min_possible
rpmin_a = 1.0 / np.nanmean(pd.to_numeric(A["lambda_selected_1yr"], errors="coerce"))
rpmin_t = 1.0 / np.nanmean(pd.to_numeric(T["lambda_selected_1yr"], errors="coerce"))
print("\nRP_min_possible (areas):", rpmin_a, "yr")
print("RP_min_possible (transects):", rpmin_t, "yr")