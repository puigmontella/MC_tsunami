import numpy as np
import pandas as pd

RHO_W = 1000.0
G = 9.81



def make_circle_arc_through_endpoints(s0, z0, s1, z1, sagitta, npts=200):
    import numpy as np

    s0 = float(s0); z0 = float(z0)
    s1 = float(s1); z1 = float(z1)
    h = float(max(sagitta, 1e-6))
    npts = int(npts)

    ds = s1 - s0
    dz = z1 - z0
    L = float(np.sqrt(ds*ds + dz*dz))
    if not np.isfinite(L) or L <= 0.0:
        return None

    r = (L*L) / (8.0*h) + 0.5*h
    if not np.isfinite(r) or r <= 0.0:
        return None

    # unit vector along chord (u) and perpendicular (v)
    u = np.array([ds, dz], dtype=float) / L
    v1 = np.array([-u[1], u[0]], dtype=float)
    v2 = -v1

    # choose v pointing "deeper": decreasing z (more negative) => v_z < 0
    v = v1 if v1[1] < v2[1] else v2

    # local coords: A=(0,0), B=(L,0), arc bulges to +y (toward v)
    x = np.linspace(0.0, L, npts)
    yc = -(r - h)
    inside = r*r - (x - 0.5*L)**2
    inside[inside < 0.0] = 0.0
    y = yc + np.sqrt(inside)  # y from 0..h

    # map back: P = A + u*x + v*y
    A = np.array([s0, z0], dtype=float)
    P = A[None, :] + x[:, None]*u[None, :] + y[:, None]*v[None, :]

    s_arc = P[:, 0]
    z_arc = P[:, 1]

    dphi = 2.0 * np.arcsin(min(1.0, L/(2.0*r)))
    return s_arc.astype(float), z_arc.astype(float), float(r), float(dphi), float(L)


def slip_on_transect(s_nodes, s_arc, z_arc):
    order = np.argsort(s_arc)
    sA = np.asarray(s_arc, float)[order]
    zA = np.asarray(z_arc, float)[order]
    return np.interp(np.asarray(s_nodes, float), sA, zA)

def kcrit_rotational(s_seg, z_bathy_seg, z_slip_seg, rho_s, su_ratio, Rur, r_circle):
    s = np.asarray(s_seg, float)
    zb = np.asarray(z_bathy_seg, float)
    zs = np.asarray(z_slip_seg, float)

    if len(s) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    ds = np.diff(s)
    if np.any(ds <= 0) or not np.all(np.isfinite(ds)):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    t = zb - zs
    if np.any(~np.isfinite(t)):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Use interval-averaged thickness (allows t=0 at endpoints)
    h = 0.5*(t[:-1] + t[1:])
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    dzs = np.diff(zs)
    dl = np.sqrt(ds*ds + dzs*dzs)
    alpha = np.abs(np.arctan2(dzs, ds))

    area = h * ds
    rho_s = float(rho_s)
    su_ratio = float(su_ratio)
    Rur = float(Rur)
    r_circle = float(r_circle)

    W  = rho_s * G * area
    Wp = (rho_s - RHO_W) * G * area

    gamma_p = (rho_s - RHO_W) * G
    sigma_v0 = gamma_p * h
    sigma_v  = (1.0 - Rur) * sigma_v0
    Su = su_ratio * sigma_v

    N = float(np.sum(Su * dl))
    A = float(np.sum(Wp * np.sin(alpha)))
    B = float(np.sum(W  * (np.cos(alpha) - h/(2.0*r_circle))))

    if not np.isfinite(N) or not np.isfinite(A) or not np.isfinite(B):
        return np.nan, np.nan, N, A, B
    if A <= 0.0 or B <= 0.0:
        return np.nan, np.nan, N, A, B

    FS0 = N / A
    kcrit = (N - A) / B
    if not np.isfinite(kcrit):
        kcrit = np.nan
    if np.isfinite(kcrit) and kcrit < 0.0:
        kcrit = 0.0

    return float(FS0), float(kcrit)

def eta0_rotational_predictive(beta_deg, T, l, d, r, dphi, rho_s, rng):
    beta = np.radians(max(float(beta_deg), 0.1))
    sinb = np.sin(beta)
    if sinb <= 0.0:
        return np.nan

    rho_s = float(rho_s)
    s = rho_s / RHO_W
    if s <= 1.0:
        return np.nan

    r = float(r); dphi = float(dphi)
    l = float(l); d = float(d); T = float(T)

    SoR = 0.5 * r * dphi
    lam = np.sqrt(max(0.0, r * d * (s + 1.0) * (s - 1.0) / s))

    U = float(rng.uniform(0.0, 1.0))
    w = (0.4 + 0.2 * U) * l

    dens_fac = (1.47 - 0.35 * (s - 1.0)) * (s - 1.0)

    val = SoR * (0.131 / sinb) * (T / l) * ((l * sinb) / d)**1.25 * (l / r)**0.63 * (dphi**0.39) * dens_fac * (w / (w + lam))

    if not np.isfinite(val):
        return np.nan
    return float(abs(val))




def sample_Rut(beta_deg, phi_deg, rng, eps_t, r_gauss_std):
    """
    Grilli-style translational pore-pressure stress-reduction factor Rut.

    Concept:
      - Compute two deterministic bounds that depend on slope beta and friction phi:
          rVt(beta)          baseline
          rt(beta, phi)      upper bound
      - Draw a random mixing parameter alpha_t and blend:
          Rut = rVt + alpha_t * (rt - rVt)

    Randomness:
      - R ~ Normal(0, r_gauss_std)
      - alpha_t = clamp(eps_t + R, -1, 1)

    Outputs are clipped so that:
      - 0 <= rVt, rt <= 0.99
      - rt >= rVt
      - 0 <= Rut <= rt

    Returns:
      Rut, rVt, rt, alpha_t, R
    """
    beta = float(beta_deg)
    phi  = float(phi_deg)

    # 1) Random draw controlling where Rut sits between rVt and rt
    R = float(rng.normal(loc=0.0, scale=float(r_gauss_std)))
    alpha_t = float(np.clip(float(eps_t) + R, -1.0, 1.0))

    # 2) Deterministic bounds (empirical formulas)
    rVt = 0.9   - 0.03  * beta
    rt  = 0.991 - 0.033 * beta + 0.0003 * phi

    # 3) Keep bounds in [0, 0.99] and enforce rt >= rVt
    rVt = float(np.clip(rVt, 0.0, 0.99))
    rt  = float(np.clip(rt,  0.0, 0.99))
    rt  = max(rt, rVt)

    # 4) Blend and enforce 0 <= Rut <= rt
    Rut = rVt + alpha_t * (rt - rVt)
    Rut = float(np.clip(Rut, 0.0, rt))

    return Rut, rVt, rt, alpha_t, R


def sample_Rut_direct(rng, Rut_max, a, b):
    """
    Direct (geometry-independent) pore-pressure weakening factor Rut in [0, Rut_max].

    - Sample x ~ Beta(a, b)  (always in [0,1])
    - Rut = Rut_max * x

    Parameters a and b control the SHAPE (not the bounds):

      - a=b=1   -> uniform in [0,1] (all values equally likely)
      - a=b=2   -> more values around 0.5 (moderate pore pressure most likely)
      - a>b     -> skewed toward 1 (larger Rut more likely)
      - a<b     -> skewed toward 0 (smaller Rut more likely
      
    Returns:
      Rut : float in [0, Rut_max]
    """

    # Keep Rut_max below 1 to avoid negative (1-Rut) in FS2
    Rut_max = min(Rut_max, 0.99)
    return Rut_max * float(rng.beta(a, b))



def sample_Rur(beta_deg, rng, eps_r, r_gauss_std):
    rr = 1.01 - 0.09 * float(beta_deg)
    rr = float(max(rr, 0.0))
    R = float(rng.normal(0.0, float(r_gauss_std)))
    Rur = R * rr + float(eps_r)
    if Rur < 0.0:
        Rur = 0.0
    if Rur > rr:
        Rur = rr
    return float(Rur), float(rr), float(R)

def sample_Rur_direct(rng, Rur_max, a, b):
    """
    Direct (geometry-independent) rotational pore-pressure weakening factor Rur in [0, Rur_max].

    - Sample x ~ Beta(a, b) in [0,1]
    - Rur = Rur_max * x

    Returns:
      Rur : float in [0, Rur_max]
    """
    Rur_max = min(float(Rur_max), 0.99)
    return Rur_max * float(rng.beta(a, b))


def kcrit_translational(beta_deg, phi_deg, rho_s, Rut):
    """
    Critical pseudo-static coefficient kcrit for translational failure using Grilli Eq.(7).

    Solve FS(k)=1 for k => kcrit.

    This form is written as Eq. (3) in Krause (2011)
    and used in the same PTHA workflow referenced by Grilli et al. (2009).

    """

    beta = np.radians(float(beta_deg))
    phi  = np.radians(float(phi_deg))

    s = float(rho_s) / RHO_W

    tb = np.tan(beta)
    tp = np.tan(phi)


    # kcrit from algebra (FS=1)
    num = tp * (s - 1.0) * (1.0 - Rut) - (s - 1.0) * tb
    den = s * (1.0 + tp * tb)

    k = num / den
    if not np.isfinite(k):
        k = np.nan
    elif k < 0.0:
        k = 0.0

    return float(k)



def eta0_translational_predictive(beta_deg, T, l, d, rho_s, w, rho_w):
    """
    Predictive empirical initial tsunami amplitude for a translational submarine slide.

    This is the Watts/Grilli-style predictive equation used in PTHA workflows:
      - it depends on slope angle beta, slide thickness T, length l, depth d, density ratio s=rho_s/rho_w,
        and includes a 3D correction w/(w+lambda0).

    """
    # --- basic checks ---
    beta = np.radians(max(float(beta_deg), 0.001))  # avoid sin(beta)=0
    sinb = np.sin(beta)

    T = float(T); l = float(l); d = float(d); w = float(w)
    if l <= 0.0 or d <= 0.0 or T <= 0.0 or w <= 0.0:
        return np.nan

    # density ratio
    s = float(rho_s) / float(rho_w)

    # --- characteristic motion scale S0 (as used in the predictive-equation framework) ---
    S0 = 0.5 * np.pi * l * (s + 1.0)

    # --- characteristic wavelength-like scale lambda0 (controls 3D width correction) ---
    lambda0 = np.sqrt((np.pi * l * d * (s + 1.0)**2) / (2.0 * sinb * (s - 1.0) * s))

    # --- fitted / geometric / density terms ---
    slope_fit = (0.0574 - 0.0431 * sinb)
    geom = (T / l) * ((l * sinb) / d)**1.25
    dens = (1.0 - np.exp(-2.2 * (s - 1.0)))

    # 3D correction: wide slides behave more 2D; narrow slides are reduced
    corr_3d = w / (w + lambda0)

    eta0 = S0 * slope_fit * geom * dens * corr_3d

    # amplitude should be non-negative
    return float(abs(eta0))



def load_transect(path):
    dfT = pd.read_csv(path)
    # find columns for distance along transect and depth, with flexible naming
    s_all = dfT["s_m"].to_numpy(float)
    z_all = dfT["depth"].to_numpy(float)

    m = np.isfinite(s_all) & np.isfinite(z_all)
    s_all = s_all[m]
    z_all = z_all[m]

    if len(s_all) < 5:
        return None, None

    order = np.argsort(s_all)
    return s_all[order], z_all[order]


def sample_thickness_T(rng, T_cfg, L_eff):
    """
    Sample slide thickness T [m]

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator  
    T_cfg : dict
        Configuration for thickness sampling. Example:
        {
          "mode": "ratio",
          "tl_min": 0.01,
          "tl_max": 0.08,
          "min_m": 5.0,
          "max_m": 200.0
        }
    L_eff : float
        Effective slide length along transect [m]  

    Returns
    -------
    float
        Sampled thickness T in meters.
    """

    mode = T_cfg["mode"]

    # Compute T depending on mode
    if mode == "ratio":
        # Pick a thickness ratio tl in [tl_min, tl_max], then T = tl * L_eff
        tl_min = float(T_cfg["tl_min"])
        tl_max = float(T_cfg["tl_max"])
        tl = float(rng.uniform(tl_min, tl_max))
        T = tl * float(L_eff)

    elif mode == "uniform":
        # Pick thickness directly in meters between min_m and max_m
        mn = float(T_cfg["min_m"])
        mx = float(T_cfg["max_m"])
        T = float(rng.uniform(mn, mx))

    elif mode == "fixed":
        # Use a constant thickness
        T = float(T_cfg["value_m"])

    else:
        raise ValueError(f"Unknown T.mode='{mode}' (use 'ratio', 'uniform', or 'fixed')")

    # Enforce  bounds
    T = max(T, float(T_cfg["min_m"]))
    T = min(T, float(T_cfg["max_m"]))

    return float(T)


def sample_bounded_lognormal(rng, median, sigma_ln, vmin, vmax, max_tries=80):
    med = float(median)
    sig = float(sigma_ln)
    lo = float(vmin)
    hi = float(vmax)

    if not (np.isfinite(med) and med > 0 and np.isfinite(sig) and sig >= 0):
        return np.nan

    mu = np.log(med)
    for _ in range(int(max_tries)):
        x = float(rng.lognormal(mean=mu, sigma=sig))
        if np.isfinite(x) and (lo <= x <= hi):
            return x

    x = float(rng.lognormal(mean=mu, sigma=sig))
    return float(np.clip(x, lo, hi)) if np.isfinite(x) else np.nan



def load_hazard_fit(hazard_dir, tid):
    path =  hazard_dir + f"/tr_{int(tid):06d}_pga_fit.npz"

    D = np.load(path)
    C = D["ln_pga_coeffs"].astype(float)

    return {
        "path": path,
        "s_m": D["s_m"].astype(float),
        "ln_coeffs": C,      # (deg+1, n_pts)
        "deg": int(D["deg"]),
    }


def eval_k_at_s(hz, s_center_m, rp_years):
    """
    Evaluate hazard k(RP) at a given transect position s_center_m.
    """
    sgrid = hz["s_m"]
    C = hz["ln_coeffs"]

    s_center_m = float(np.clip(float(s_center_m), float(sgrid[0]), float(sgrid[-1])))
    c_at_s = np.array([np.interp(s_center_m, sgrid, C[i, :]) for i in range(C.shape[0])], dtype=float)

    rp = np.asarray(rp_years, dtype=float)
    prob = 1.0 / rp

    lnval = np.polyval(c_at_s, prob)
    k = np.exp(lnval)
    k = np.maximum.accumulate(k)  # enforce monotonic in RP
    return k


def FS_translational(beta_deg, phi_deg, rho_s, Rut, k):
    beta = np.radians(float(beta_deg))
    phi = np.radians(float(phi_deg))
    Ru = float(Rut)
    s = float(rho_s) / RHO_W

    tb = np.tan(beta)
    tp = np.tan(phi)
    k = float(k)

    den = ((s - 1.0) * tb + k * s)
    if not np.isfinite(den) or den == 0.0:
        return np.nan

    num = ((s - 1.0) * (1.0 - Ru) - k * s * tb)
    return float((num / den) * tp)


# -----------------------------
# Newmark model
# -----------------------------
def newmark_disp_rathje_saygili_2009_m(PGA_g, critaccel_g, M, eps_ln=0.0):
    PGA_g = np.asarray(PGA_g, dtype=float)
    crit = float(critaccel_g)
    M = float(M)

    D = np.zeros_like(PGA_g, dtype=float)
    m = np.isfinite(PGA_g) & (PGA_g > 0.0) & (PGA_g > crit) & np.isfinite(crit) & (crit > 0.0)
    if not np.any(m):
        return D

    r = crit / PGA_g[m]
    lnD_cm = (
        4.89
        - 4.85 * r
        - 19.64 * r**2
        + 42.49 * r**3
        - 29.06 * r**4
        + 0.72 * np.log(PGA_g[m])
        + 0.89 * (M - 6.0)
        + float(eps_ln)
    )
    D[m] = np.exp(lnD_cm) / 100.0
    return D

