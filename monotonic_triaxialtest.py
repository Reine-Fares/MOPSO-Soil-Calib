"""
Some important reference pages and articles 
https://github.com/OpenSees
Zhu, M., McKenna, F., & Scott, M. H. (2018). OpenSeesPy: Python library for the OpenSees finite element framework. SoftwareX, 7, 6-11.
https://opensees.github.io/OpenSeesDocumentation/user/manual/material/ndMaterials/ManzariDafalias.html
Dafalias YF, Manzari MT. “Simple plasticity sand model accounting for fabric change effects”. Journal of Engineering Mechanics 2004
"""

import math
from pathlib import Path
import numpy as np



# -------------------------------------------------------------------
# Monotonic triaxial test 
# -------------------------------------------------------------------

def run_monotonic_triaxial(soil_params, devDisp=-0.17):
    import opensees as ops #if opensees github
    # import openseespy.opensees as ops #if opensees package python
    """
    Return:
      - convergence (0 if ok)
      - eps : Evolution of strain e(t)
      - q : Evolution of deviatoric stress q(t)

    """
    G0 = soil_params["G0"]
    nu = soil_params["nu"]
    Mc = soil_params["Mc"]
    c = soil_params["c"]
    lambda_c = soil_params["lambda_c"]
    e0_MZ = soil_params["e0"]
    ksi = soil_params["ksi"]
    P_atm = soil_params["P_atm"]
    m_md = soil_params["m"]
    h0 = soil_params["h0"]
    ch = soil_params["ch"]
    nb = soil_params["nb"]
    A0 = soil_params["A0"]
    nd = soil_params["nd"]
    z_max = soil_params["z_max"]
    cz = soil_params["cz"]
    Den = soil_params["Den"]
    pConf = soil_params["pConf"]
    perm = soil_params["perm"]
    e_init = soil_params["vR"]
    br = 2.2e6 

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 4)
    # ops.logFile("screen.txt", "-noEcho")
    ops.logFile("/dev/null", "-noEcho")
    ops.node(1, 1, 0, 0)
    ops.node(2, 1, 1, 0)
    ops.node(3, 0, 1, 0)
    ops.node(4, 0, 0, 0)
    ops.node(5, 1, 0, 1)
    ops.node(6, 1, 1, 1)
    ops.node(7, 0, 1, 1)
    ops.node(8, 0, 0, 1)
    ops.fix(1, 0, 1, 1, 1)
    ops.fix(2, 0, 0, 1, 1)
    ops.fix(3, 1, 0, 1, 1)
    ops.fix(4, 1, 1, 1, 1)
    ops.fix(5, 0, 1, 0, 1)
    ops.fix(6, 0, 0, 0, 1)
    ops.fix(7, 1, 0, 0, 1)
    ops.fix(8, 1, 1, 0, 1)
    ops.nDMaterial(
        "ManzariDafalias",
        1,
        G0,
        nu,
        e_init,
        Mc,
        c,
        lambda_c,
        e0_MZ,
        ksi,
        P_atm,
        m_md,
        h0,
        ch,
        nb,
        A0,
        nd,
        z_max,
        cz,
        Den,
    )
    ops.element(
        "SSPbrickUP",
        1,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        1,
        br,
        1.0,
        perm,
        perm,
        perm,
        e_init,
        1.5e-9,
    )

    ops.constraints("Penalty", 1e18, 1e18)
    ops.test("NormDispIncr", 1e-5, 20, 1)
    ops.algorithm("Newton")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    pNode = pConf / 4.0
    ops.timeSeries("Path", 1, "-values", 0, 1, 1, "-time", 0, 1e4, 1e10, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    ops.load(1, pNode, 0, 0, 0)
    ops.load(2, pNode, pNode, 0, 0)
    ops.load(3, 0, pNode, 0, 0)
    ops.load(4, 0, 0, 0, 0)
    ops.load(5, pNode, 0, pNode, 0)
    ops.load(6, pNode, pNode, pNode, 0)
    ops.load(7, 0, pNode, pNode, 0)
    ops.load(8, 0, 0, pNode, 0)
    ops.analyze(100, 100.0)
    ops.analyze(100, 100.0)

    # for nd in range(1, 9):
        # ops.remove("sp", nd, 4)
    # ops.analyze(5, 0.1)
    ops.updateMaterialStage("-material", 1, "-stage", 1)
    ops.analyze(10, 0.1)

    vertDisp = ops.nodeDisp(5, 3)
    if abs(vertDisp) < 1e-12:
        vertDisp = -1e-12
    ops.timeSeries(
        "Path",
        5,
        "-values",
        1,
        1 + (devDisp / vertDisp),
        1 + (devDisp / vertDisp),
        "-time",
        20001,
        1020000,
        10020000,
        "-factor",
        1.0,
    )
    ops.pattern("Plain", 2, 5)
    for jnd in (5, 6, 7, 8):
        ops.sp(jnd, 3, vertDisp)

    eps1_list = []
    q_list = []


    dT = 20
    nSteps = 50000
    s0 = np.asarray(ops.eleResponse(1, "stress"), float)
    e0 = np.asarray(ops.eleResponse(1, "strain"), float)
    if s0.size < 3 or e0.size < 3:
        ops.wipe()
        return np.array([]), np.array([])
    SIGX0 = -s0[0]
    SIGZ0 = -s0[2]
    SIG0 = SIGX0 - abs(pConf)
    for _ in range(nSteps):
        ok = ops.analyze(1, dT)
        tries = 0
        while ok != 0 and dT > 1e-5 and tries < 8:
            dT *= 0.5
            ok = ops.analyze(1, dT)
            tries += 1
        if ok != 0:
            break
        s = np.asarray(ops.eleResponse(1, "stress"), float)
        e = np.asarray(ops.eleResponse(1, "strain"), float)
        if s.size < 3 or e.size < 3:
            break
        SIGX = -s[0] - SIG0
        SIGZ = -s[2] - SIG0
        q = SIGZ - SIGX
        eps1 = -(e[2] - e0[2]) * 100.0
        eps1_list.append(eps1)
        q_list.append(q)
    ops.wipe()
    return (ok,
    np.asarray(eps1_list, float),
    np.asarray(q_list, float)
    )

