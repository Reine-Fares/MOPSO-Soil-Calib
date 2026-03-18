"""
Soil Model Parameters
=====================

This file defines:

    - Base material parameters of the constitutive model (Manzari-Dafalias for this exemple)
    - Solver time-step parameters
    - Calibrated parameters and their bounds
    - Parameter discretization steps (snap-to-grid)

Some parameters remain fixed while others are calibrated during the PSO
optimization procedure.
"""


import os
import math
import itertools
import numpy as np


# =============================================================================
# Base material parameters (fixed values)
# =============================================================================
materiau_base = dict(
    G0=176.0,       # shear modulus constant
    nu=0.05,        # poisson ratio
    Mc=1.37,        # critical state stress ratio
    c=0.712,        # ratio of critical state stress ratio in extension and compression
    P_atm=100.0,    # !! atmospheric pressure
    m=0.014,        # yield surface constant (radius of yield surface in stress ratio space)
    Den=1.94,       # mass density of the material
    lambda_c=0.065, # critical state line constant
    ksi=0.40,       # critical state line constant
    nb=0.60,        # bounding surface parameter $nb ≥ 0
    nd=2.5,         # dilatancy surface parameter $nd ≥ 0
    h0=7.224,       # constant parameter
    ch=1.052,       # constant parameter
    A0=0.087,       # dilatancy parameter
    z_max=76.363,   # fabric-dilatancy tensor parameter
    cz=511.838,     # fabric-dilatancy tensor parameter
    pConf=-200.0,   # confinement pressure
    perm=5.0e-4,    # permiability
    e0=1.0,         # critical void ratio at p = 0
    vR=0.7592,      # initial void ratio
)


# =============================================================================
# Calibrated parameters and bounds
# =============================================================================

PARAM_NAMES = ["h0", "ch", "A0", "z_max", "cz"]         # List of parameters to calibrate
LOW = np.array([2.0, 0.30, 0.05, 10.0, 400.0], float)   # Lower bounds of the search space
UP = np.array([30.0, 1.30, 1.00, 100.0, 800.0], float)  # Upper bounds of the search space
BOUNDS = (LOW, UP)  # Tuple used by the optimizer
PENALTY = 1e12      # !! Applied penalty 

STEP = np.array([
    0.1,    # h0
    0.001,  # ch
    0.001,  # A0
    1.0,    # z_max
    1.0,    # cz
], dtype=float) # Parameter discretization (snap-to-grid resolution)

# =============================================================================
# Solver parameters
# =============================================================================

FREQ_HZ = 0.10 # Loading frequency in Hz
CSR = 0.22 # Cyclic stress ratio
DEVDISP = -0.17 # Deviatoric displacement
period = 1.0 / max(FREQ_HZ, 1e-12) # Period
CYCNUM = 15 # Number of loading cycles
dT=0.01     # Initial time step (recommended value for repeatability is 0.001)
dTmin=0.001 # Minimum allowed time step (recommended value for repeatability is 0.0001)
