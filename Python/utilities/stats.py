
import os
import re
import numpy as np



def quantile_safe(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.quantile(a, p))

def weighted_percentile(R, w, q):
    """
    Weighted percentile of R with weights w.

    Parameters
    ----------
    R : array-like
        Values.
    w : array-like
        Non-negative weights (only w>0 are kept).
    q : float
        Percentile in [0, 100].

    Returns
    -------
    float
        Weighted percentile value (np.nan if no valid data).
    """
    R = np.asarray(R, float)
    w = np.asarray(w, float)

    m = np.isfinite(R) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return np.nan

    Rm = R[m]
    wm = w[m]

    # sort by R
    idx = np.argsort(Rm)
    Rm = Rm[idx]
    wm = wm[idx]

    # normalized cumulative weights
    cw = np.cumsum(wm)
    cw /= cw[-1]

    # interpolate percentile position
    return float(np.interp(float(q) / 100.0, cw, Rm))

def _valid_mask(*arrs):
    m = None
    for a in arrs:
        a = np.asarray(a)
        mi = np.isfinite(a)
        m = mi if m is None else (m & mi)
    return m

def prob_leq(x, k):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if not np.any(m):
        return np.nan
    return float(np.mean(x[m] <= k))

def prob_joint_leq_geq(kcrit, k, eta0, h):
    kcrit = np.asarray(kcrit, float)
    eta0 = np.asarray(eta0, float)
    m = _valid_mask(kcrit, eta0)
    if not np.any(m):
        return np.nan
    return float(np.mean((kcrit[m] <= k) & (eta0[m] >= h)))

# -------------------------
# Sampling helpers (priors)
# -------------------------
def lognormal_mu_sigma_from_mean_std(mean, std):
    mean = float(mean)
    std = float(std)
    var = std * std
    sigma2 = np.log(1.0 + var / (mean * mean))
    mu = np.log(mean) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return float(mu), float(sigma)

def trunc_normal(rng, mean, std, vmin, vmax, size=None):
    mean = float(mean); std = float(std)
    vmin = float(vmin); vmax = float(vmax)

    if size is None:
        while True:
            x = float(rng.normal(mean, std))
            if vmin <= x <= vmax:
                return x

    out = np.empty(int(size), dtype=float)
    i = 0
    while i < out.size:
        batch = rng.normal(mean, std, size=(out.size - i) * 2)
        batch = batch[(batch >= vmin) & (batch <= vmax)]
        n = min(batch.size, out.size - i)
        if n > 0:
            out[i:i+n] = batch[:n]
            i += n
    return out

def trunc_lognormal(rng, mean, std, vmin, vmax, size=None):
    mean = float(mean); std = float(std)
    vmin = float(vmin); vmax = float(vmax)
    mu, sigma = lognormal_mu_sigma_from_mean_std(mean, std)

    if size is None:
        while True:
            x = float(rng.lognormal(mu, sigma))
            if vmin <= x <= vmax:
                return x

    out = np.empty(int(size), dtype=float)
    i = 0
    while i < out.size:
        batch = rng.lognormal(mu, sigma, size=(out.size - i) * 2)
        batch = batch[(batch >= vmin) & (batch <= vmax)]
        n = min(batch.size, out.size - i)
        if n > 0:
            out[i:i+n] = batch[:n]
            i += n
    return out

# -------------------------
# Transect helpers
# -------------------------
def parse_transect_id(path):
    base = os.path.basename(path)
    m = re.search(r"tr_(\d{4})", base)
    if m:
        return int(m.group(1))
    m = re.search(r"tr_(\d+)", base)
    if not m:
        raise ValueError("Could not parse transect id from filename: {}".format(base))
    return int(m.group(1))


def beta_from_segment(s_seg, depth_pos_seg):
    s = np.asarray(s_seg, float)
    d = np.asarray(depth_pos_seg, float)
    m = np.isfinite(s) & np.isfinite(d)
    if m.sum() < 2:
        return np.nan
    b = np.polyfit(s[m], d[m], 1)[0]
    return float(np.degrees(np.arctan(abs(b))))

# -------------------------
# Formatting helpers (optional)
# -------------------------
def fmt_k(k):
    return "k{:03d}".format(int(round(float(k) * 1000)))

def fmt_h(h):
    s = "{:.1f}".format(float(h)).replace(".", "p")
    return "h{}".format(s)
