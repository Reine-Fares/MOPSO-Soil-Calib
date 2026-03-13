import os, math, itertools
import math
import numpy as np

"""
Cost Functions
==============

This file defines the cost functions used in the calibration procedure.

It includes:
    - generic RMSE-based metrics
    - cyclic cost functions
    - monotonic cost functions
    - helper functions for threshold/cycle detection
    - helper functions for yield-point estimation
These functions are used to compare simulated and experimental responses during optimization.
"""


# Compute the Root Mean Square Error (RMSE) between two arrays.
def rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return math.inf
    return float(np.sqrt(np.mean((a - b) ** 2)))

# Compute the normalized RMSE between two curves 
def rmse_on_grid(xE, yE, xS, yS, n=400):
    xE = np.asarray(xE, float)
    yE = np.asarray(yE, float)
    xS = np.asarray(xS, float)
    yS = np.asarray(yS, float)
    if xE.size < 2 or xS.size < 2:
        return math.inf
    xmin = min(xE)
    xmax = max(xE)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return math.inf
    xg = np.linspace(xmin, xmax, n)
    yEi = np.interp(xg, xE, yE)
    ySi = np.interp(xg, xS, yS)
    den = max(np.ptp(yEi), 1e-9)
    return float(np.sqrt(np.mean((ySi - yEi) ** 2)) / den)

# Compute a weighted normalized RMSE on the overlapping range of two curves
def rmse_on_grid_p(xE, yE, xS, yS, n=500):
    xE = np.asarray(xE, float)
    yE = np.asarray(yE, float)
    xS = np.asarray(xS, float)
    yS = np.asarray(yS, float)
    if xE.size < 2 or xS.size < 2:
        return math.inf
    xmin = max(float(np.min(xE)), float(np.min(xS)))
    xmax = min(float(np.max(xE)), float(np.max(xS)))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return math.inf
    xg = np.linspace(xmin, xmax, n)
    yEi = np.interp(xg, xE, yE)
    ySi = np.interp(xg, xS, yS)
    den = max(np.ptp(yEi), 1e-9)
    value = 0.0
    value += (rmse(yEi[1:201],   ySi[1:201])   / den) * 0.10
    value += (rmse(yEi[201:301], ySi[201:301]) / den) * 0.25
    value += (rmse(yEi[301:401], ySi[301:401]) / den) * 0.30
    value += (rmse(yEi[401:501], ySi[401:501]) / den) * 0.35
    return value


# Estimate the number of cycles required to reach a displacement threshold
def cycles_to_threshold(N, u, uth):
    N = np.asarray(N, float)
    u = np.asarray(u, float)
    if N.size < 2 or u.size < 2:
        return np.inf
    idx = np.where(u >= uth)[0]
    if idx.size == 0:
        return np.inf
    k = int(idx[0])
    if k == 0:
        return float(N[0])
    u0, u1 = float(u[k - 1]), float(u[k])
    N0, N1 = float(N[k - 1]), float(N[k])
    if u1 == u0:
        return N1
    t = (uth - u0) / (u1 - u0)
    return float(N0 + t * (N1 - N0))

# Relative error in the number of cycles needed to reach a threshold defined by theta * |pConf|
def cost_cyc_Ntheta(Ns, Us, Ne, Ue, pConf, theta, big=1e6):
    uth = float(theta) * abs(pConf)
    Nexp = cycles_to_threshold(Ne, Ue, uth)
    Nsim = cycles_to_threshold(Ns, Us, uth)
    if np.isfinite(Nexp) and np.isfinite(Nsim):
        return abs(Nsim - Nexp) / max(Nexp, 1e-9)
    return big


# Return the experimental cycle count corresponding to a threshold theta
def Nref_N80(Ne, Ue, pConf, theta):
    uth = float(theta) * abs(pConf)
    Nexp = cycles_to_threshold(Ne, Ue, uth)
    return Nexp
    

# Average cyclic threshold-based cost computed for theta = 0.7, 0.8, and 0.9
def cost_cyc_avg3(Ns, Us, Ne, Ue, pConf):
    return (
        cost_cyc_Ntheta(Ns, Us, Ne, Ue, pConf, 0.7)
        + cost_cyc_Ntheta(Ns, Us, Ne, Ue, pConf, 0.8)
        + cost_cyc_Ntheta(Ns, Us, Ne, Ue, pConf, 0.9)
    ) / 3.0

# Cyclic cost based on normalized RMSE between experimental and simulated
def cost_cyc_rmse(Ns, Us, Ne, Ue):
    return rmse_on_grid(Ne, Ue, Ns, Us, n=300)

# Combined cyclic cost based on:
# - RMSE over the full cyclic curve
# - N80 threshold-cycle error
def cost_N80(Ns,  Us, Ne, Ue, pconf):
    J_u = cost_cyc_rmse(Ns, Us, Ne, Ue)
    J_80 = cost_cyc_Ntheta(Ns, Us, Ne, Ue, pConf, 0.8)
    
    return (J_u + J_80)/2.0 
    
# Weighted cyclic RMSE on overlapping interpolation range
def cost_u_rmse_u(Ns, Us, Ne, Ue, n=500) :
    return rmse_on_grid_p (Ns, Us, Ne, Ue)     



# Monotonic cost based on normalized RMSE between q-epsilon curves
def cost_mono_qeps(epsS, qS, eps_exp, q_exp):
    return rmse_on_grid(epsS, qS, eps_exp, q_exp , n=400) 

# Estimate the bilinear breakpoint (approximate yield point) of a curve
def bilinear_breakpoint(eps, q, n_candidates=60):
    eps = np.asarray(eps, float)
    q = np.asarray(q, float)
    n = len(eps)
    if n < 20:
        return float("nan"), float("nan")
    idxs = np.linspace(5, n - 5, n_candidates, dtype=int)
    best = (eps[1], 0.0, 0.0, 1e18, eps[1], q[1])
    for k in idxs:
        a1, b1 = np.polyfit(eps[:k], q[:k], 1)
        a2, b2 = np.polyfit(eps[k:], q[k:], 1)
        qhat = np.r_[a1 * eps[:k] + b1, a2 * eps[k:] + b2]
        err = float(np.mean((qhat - q) ** 2))
        ey = float(eps[k])
        qy = float(a1 * ey + b1)
        if err < best[3]:
            best = (a1, b1, a2, b2, ey, qy)
    return best[4], best[5]

# Cost based on the mismatch in the estimated yield point
def cost_mono_yield(epsS, qS, eps_exp, q_exp):
    try:
        eyE, qyE = bilinear_breakpoint(eps_exp, q_exp)
        eyS, qyS = bilinear_breakpoint(epsS, qS)
        if not (
            np.isfinite(eyE)
            and np.isfinite(qyE)
            and np.isfinite(eyS)
            and np.isfinite(qyS)
        ):
            return cost_mono_qeps(epsS, qS, eps_exp, q_exp)
        t_eps = abs(eyS - eyE) / max(abs(eyE), 1e-9)
        t_q = abs(qyS - qyE) / max(abs(qyE), 1e-6)
        return 0.7 * t_eps + 0.3 * t_q
    except Exception:
        return cost_mono_qeps(epsS, qS, eps_exp, q_exp)

# Cost based on the stress value at the maximum experimental strain
def cost_mono_qmax(epsS, qS):
    emax = float(np.max(eps_exp))
    qE = float(q_exp[np.argmax(eps_exp)])
    qSi = float(
        np.interp(
            emax,
            np.asarray(epsS, float),
            np.asarray(qS, float),
            left=float(qS[0]),
            right=float(qS[-1]),
        )
    )
    return abs(qSi - qE) / max(abs(qE), 1e-6)

# projects parameter values onto a discretized grid while enforcing parameter bounds
def snap_to_grid(x, low, up, step):

    x = np.asarray(x, float)
    y = low + np.round((x - low) / step) * step
    return np.clip(y, low, up)

