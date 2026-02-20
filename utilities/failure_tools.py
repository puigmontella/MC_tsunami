import numpy as np

RHO_W = 1000.0
G = 9.81

def sample_Rur(beta_deg, rng, eps_r=0.35, r_gauss_std=0.25):
    rr = 1.01 - 0.09 * float(beta_deg)
    rr = float(max(rr, 0.0))
    R = float(rng.normal(0.0, float(r_gauss_std)))
    Rur = R * rr + float(eps_r)
    if Rur < 0.0:
        Rur = 0.0
    if Rur > rr:
        Rur = rr
    return float(Rur), float(rr), float(R)


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

def rotational_fs0_kcrit_eq10(s_seg, z_bathy_seg, z_slip_seg, rho_s, su_ratio, Rur, r_circle):
    s = np.asarray(s_seg, float)
    zb = np.asarray(z_bathy_seg, float)
    zs = np.asarray(z_slip_seg, float)

    if len(s) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    ds = np.diff(s)
    if np.any(ds <= 0) or not np.all(np.isfinite(ds)):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    t = zb - zs
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    h = 0.5*(t[:-1] + t[1:])
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

    return float(FS0), float(kcrit), N, A, B

def eta0_rotational_eq19(beta_deg, T, l, d, r, dphi, rho_s, rng):
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





def sample_Rut(beta_deg, phi_deg, rng, eps_t=0.5, r_gauss_std=0.25):
    """
    Grilli et al. (2009) translational pore pressure stress-reduction factor Rut.

    Eq.(13): Rut = rVt + alpha_t (rt - rVt)
      alpha_t = R + eps_t, clamped to [-1, 1]
      R ~ N(0, r_gauss_std)

    Eq.(14): rVt = 0.9 - 0.03 beta
    Eq.(15): rt  = 0.991 - 0.033 beta + 0.0003 phi
    """
    beta = float(beta_deg)
    phi  = float(phi_deg)

    R = float(rng.normal(0.0, float(r_gauss_std)))
    alpha_t = float(R + float(eps_t))
    if alpha_t < -1.0:
        alpha_t = -1.0
    if alpha_t > 1.0:
        alpha_t = 1.0

    rVt = 0.9 - 0.03 * beta
    rt  = 0.991 - 0.033 * beta + 0.0003 * phi

    # keep reasonable bounds
    rVt = float(np.clip(rVt, 0.0, 0.99))
    rt  = float(np.clip(rt,  0.0, 0.99))

    Rut = rVt + alpha_t * (rt - rVt)
    Rut = float(np.clip(Rut, 0.0, rt))

    return Rut, rVt, rt, alpha_t, R


def kcrit_translational(beta_deg, phi_deg, rho_s, Rut):
    """
    Critical pseudo-static coefficient kcrit for translational failure using Grilli Eq.(7).

    Interpreting Eq.(7) as:
      FS = tan(phi) * [ (s-1)(1-Ru) - k*s*tan(beta) ] / [ (s-1)tan(beta) + k*s ]
    Solve for FS=1 => kcrit.

    Returns kcrit (>=0) and FS0 (k=0).
    """
    beta = np.radians(float(beta_deg))
    phi  = np.radians(float(phi_deg))
    Ru   = float(Rut)

    s = float(rho_s) / RHO_W
    if not np.isfinite(s) or s <= 1.0:
        return np.nan, np.nan

    tb = np.tan(beta)
    tp = np.tan(phi)

    # FS at k=0
    if tb <= 0.0:
        FS0 = np.inf
    else:
        FS0 = (1.0 - Ru) * tp / tb

    # kcrit from algebra (FS=1)
    num = tp * (s - 1.0) * (1.0 - Ru) - (s - 1.0) * tb
    den = s * (1.0 + tp * tb)

    if den <= 0.0 or not np.isfinite(num) or not np.isfinite(den):
        return np.nan, FS0

    k = num / den
    if not np.isfinite(k):
        k = np.nan
    elif k < 0.0:
        k = 0.0

    return float(k), float(FS0)


def eta0_translational_eq18(beta_deg, T, l, d, rho_s, w):
    """
    Initial tsunami characteristic amplitude for translational slides (Grilli Eq.(18)).
    Uses:
      SoT (Eq.20): SoT = (pi/2) * l * (s + 1)
      lambda_oT (Eq.22): sqrt( pi*l*d*(s+1)^2 / (2*sin(beta)*(s-1)*s) )

      eta0T = SoT*(0.0574 - 0.0431*sin(beta))*(T/l) * ((l*sin(beta))/d)^1.25
              * (1 - exp(-2.2*(s-1))) * w/(w+lambda_oT)

    Returns eta0 (>=0).
    """
    beta = np.radians(max(float(beta_deg), 0.1))
    sinb = np.sin(beta)
    if sinb <= 0.0:
        return np.nan

    T = float(T); l = float(l); d = float(d); w = float(w)
    if l <= 0.0 or d <= 0.0 or T <= 0.0 or w <= 0.0:
        return np.nan

    s = float(rho_s) / RHO_W
    if s <= 1.0:
        return np.nan

    SoT = 0.5 * np.pi * l * (s + 1.0)
    lam = np.sqrt((np.pi * l * d * (s + 1.0)**2) / (2.0 * sinb * (s - 1.0) * s))

    pref = (0.0574 - 0.0431 * sinb)
    geom = (T / l) * ((l * sinb) / d)**1.25
    dens = (1.0 - np.exp(-2.2 * (s - 1.0)))

    eta0 = SoT * pref * geom * dens * (w / (w + lam))
    if not np.isfinite(eta0):
        return np.nan
    return float(abs(eta0))
