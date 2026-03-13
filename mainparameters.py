"""
Main Parameters
===============

This file defines the input parameters used for the model calibration.

It includes:
    - plotting options
    - output directory configuration
    - loss function parameters
    - selected weights used in the loss function
      (the authors examined several beta-weight configurations and recommend
       beta1 = 1 and beta2 = 0)
    - PSO configuration parameters
    - solver parameters
"""

from pathlib import Path

# =============================================================================
# General options
# =============================================================================

# Plotting option
# 0 = yes
# 1 = no
ploting = 0

# =============================================================================
# Input / Output
# =============================================================================

# Base directory of the current file
BASE = Path(__file__).parent.resolve()

# Root output directory
OUTROOT = BASE / "sortie"
OUTROOT.mkdir(parents=True, exist_ok=True)

# ===========================
# Parameters
# ===========================

# -----------------------------------------------------------------
# Loss function parameters
# -----------------------------------------------------------------

# Weights used in the loss function
BETA1 = 1.0 # !!
BETA2 = 0.0 # !!

# Selected cost functions
CYCLIC_COSTS = ["rmse"]     # !!
MONOTONIC_COSTS = ["nrmse"] # !!

# Alpha configuration
# If ALPHAS contains several values, the script performs an alpha scan.
# If ALPHAS contains a single value, the script performs a direct calibration.
# ALPHAS = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ALPHAS = [-0.6]



# -----------------------------------------------------------------
# PSO parameters
# -----------------------------------------------------------------

N_PART = 30 # Number of particles
N_ITERS = 20 # Number of PSO iterations
SEED=123 # !! Random seed for reproducibility
VFRAC=0.15 # !! Velocity fraction


# -----------------------------------------------------------------
# Solver parameters
# -----------------------------------------------------------------

FREQ_HZ = 0.10 # Loading frequency in Hz
CSR = 0.22 # Cyclic stress ratio
DEVDISP = -0.17 # Deviatoric displacement
period = 1.0 / max(FREQ_HZ, 1e-12) # Period
CYCNUM = 15 # Number of loading cycles