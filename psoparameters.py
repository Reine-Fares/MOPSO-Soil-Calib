"""
PSO Parameter Configuration
===========================

This file defines the parameters controlling the Particle Swarm Optimization (PSO)
used for model calibration.

It includes:
    - PSO restart strategy parameters
    - adaptive PSO coefficients
    - reference discretization parameters
"""




# =============================================================================
# PSO restart strategy
# =============================================================================

STAG_ITERS = 5              # !! Number of iterations without improvement before triggering a restart
RESTART_FRAC = 0.15         # !! Fraction of particles that will be reinitialized during a restart
RESTART_SIGMA = 0.15        # !! Noise amplitude used when reinitializing particles expressed as a fraction of the search space (upper - lower bounds)
RESTART_GLOBAL_FRAC = 0.30  # !! Fraction of restarted particles initialized using global exploration


# =============================================================================
# PSO parameters
# =============================================================================

chi = 0.729                 # !! Constriction coefficient used in PSO velocity update
N_PART = 30 # Number of particles
N_ITERS = 20 # Number of PSO iterations
SEED=123 # !! Random seed for reproducibility
VFRAC=0.15 # !! Velocity fraction

# =============================================================================
# warm-up phase
# =============================================================================

n_ref = 10 # !! Number of particles 

# =============================================================================
# Adaptive PSO coefficients
# =============================================================================
w_max, w_min = 0.9, 0.6     # !! Inertia weight range controls exploration vs exploitation balance
c1_max, c1_min = 2.5, 0.5   # !! Cognitive coefficient range (particle self-attraction)
c2_min, c2_max = 0.5, 2.5   # !! Social coefficient range (swarm attraction to global best)
