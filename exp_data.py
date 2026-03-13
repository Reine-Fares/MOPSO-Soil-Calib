from pathlib import Path
import pandas as pd
import numpy as np


"""
Experimental Data Loader
========================

This module loads the experimental data used during calibration.

Two datasets are required:

    - Cyclic test data
        Columns: N (number of cycles), u (absolute displacement)

    - Monotonic test data
        Columns: eps1 (%) or strain, q (deviatoric stress)

The functions return sorted arrays and amplitude values used for normalization in cost functions.
"""


# =============================================================================
# Experimental data file paths
# =============================================================================
# Cyclic test data
# Format: N , u (absolute displacement)
EXP_CYC_FILE = BASE / "donnees_exp/Cyc200.txt"  
# Monotonic test data
# Format: eps1[%] , q[kPa]
EXP_MONO_FILE = BASE / "donnees_exp/MonoCD200.txt"  


# =============================================================================
# Cyclic experimental data
# =============================================================================

def load_exp_cyc(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, header=None, comment="#")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="any")
    if df.shape[1] < 2:
        raise RuntimeError("Cyc: besoin >=2 colonnes")
    N = df.iloc[:, 0].to_numpy(float)
    u = df.iloc[:, 1].to_numpy(float)
    idx = np.argsort(N)
    return N[idx], u[idx], max(np.ptp(u), 1e-9)


# =============================================================================
# Monotonic experimental data
# =============================================================================

def load_exp_mono(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, header=None, comment="#")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="any")
    if df.shape[1] < 2:
        raise RuntimeError("Mono: besoin >=2 colonnes")
    eps = df.iloc[:, 0].to_numpy(float)
    q = df.iloc[:, 1].to_numpy(float)
    idx = np.argsort(eps)
    return eps[idx], q[idx], max(np.ptp(q), 1e-9)


N_exp, u_exp, amp_u = load_exp_cyc(EXP_CYC_FILE)
eps_exp, q_exp, amp_q = load_exp_mono(EXP_MONO_FILE)

