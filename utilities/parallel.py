from joblib import Parallel, delayed

def seed_for_transect(global_seed, transect_id):
    return int(global_seed) + int(transect_id)

def run_parallel(items, func, n_jobs=-1, backend="loky"):
    return Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(func)(it) for it in items
    )
