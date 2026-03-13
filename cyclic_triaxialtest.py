"""
Some important reference pages and articles 
https://github.com/OpenSees
Zhu, M., McKenna, F., & Scott, M. H. (2018). OpenSeesPy: Python library for the OpenSees finite element framework. SoftwareX, 7, 6-11.
https://opensees.github.io/OpenSeesDocumentation/user/manual/material/ndMaterials/ManzariDafalias.html
Dafalias YF, Manzari MT. “Simple plasticity sand model accounting for fabric change effects”. Journal of Engineering Mechanics 2004
"""
import os, math, itertools
import numpy as np



# -------------------------------------------------------------------
# Simulation cyclique OpenSeesPy 
# -------------------------------------------------------------------
def run_cyclic_triaxial(soil_params, period, cycNum, Tcc, dT=0.01, dTmin=0.0001):
    """
    Return:
      - convergence (0 if ok)
      - Ns : Evolution of number of cycles
      - Us : Evolution of excess per water pressure
      - ps : Evolution of mean effective stress p(t)
      - qs : Evolution of deviatoric stress q(t)

    """
    import opensees as ops #si opensees github
    # import openseespy.opensees as ops #si opensees package python

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

    n_por = max(1e-6, e_init / (1.0 + e_init))
    bulk_f = 2.2e6 / n_por
    q_amp = 2.0 * pConf * Tcc

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 4)
    # ops.logFile("screen.txt", "-noEcho")
    ops.logFile("/dev/null", "-noEcho")
    # nodes
    ops.node(1, 1, 0, 0); ops.node(2, 1, 1, 0); ops.node(3, 0, 1, 0); ops.node(4, 0, 0, 0)
    ops.node(5, 1, 0, 1); ops.node(6, 1, 1, 1); ops.node(7, 0, 1, 1); ops.node(8, 0, 0, 1)
    # bcs
    ops.fix(1, 0, 1, 1, 1); ops.fix(2, 0, 0, 1, 1); ops.fix(3, 1, 0, 1, 1); ops.fix(4, 1, 1, 1, 1)
    ops.fix(5, 0, 1, 0, 1); ops.fix(6, 0, 0, 0, 1); ops.fix(7, 1, 0, 0, 1); ops.fix(8, 1, 1, 0, 1)

    ops.nDMaterial(
        "ManzariDafalias", 1,
        G0, nu, e_init, Mc, c, lambda_c, e0_MZ, ksi, P_atm, m_md,
        h0, ch, nb, A0, nd, z_max, cz, Den
    )
    ops.element(
        "SSPbrickUP", 1,
        1,2,3,4,5,6,7,8,
        1, bulk_f, 1.0, perm,perm,perm, e_init, 1.5e-9
    )

    ops.constraints("Penalty", 1e18, 1e18)
    ops.test("NormDispIncr", 1e-5, 25, 1)
    ops.algorithm("Newton")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    
    # Rayleigh damping parameter
    # damp  = 0.1
    # omega1 = 0.01
    # omega2 = 50.0
    # a1 = 2.0*damp/(omega1+omega2)
    # a0 = a1*omega1*omega2
    
    # ops.rayleigh(a0, 0., a1, 0.0)  

    # confinement
    pNode = pConf / 4.0
    ops.timeSeries("Path", 1, "-values", 0, 1, 1, "-time", 0, 1e4, 1e10, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    ops.load(1, pNode, 0, 0, 0); ops.load(2, pNode, pNode, 0, 0); ops.load(3, 0, pNode, 0, 0); ops.load(4, 0, 0, 0, 0)
    ops.load(5, pNode, 0, pNode, 0); ops.load(6, pNode, pNode, pNode, 0); ops.load(7, 0, pNode, pNode, 0); ops.load(8, 0, 0, pNode, 0)
    ops.analyze(100, 100.0); ops.analyze(50, 100.0)
    for nd_ in range(1, 9):
        ops.remove("sp", nd_, 4)
    ops.analyze(50, 100.0)

    # cyclic deviatoric load
    t0 = 20000.0
    ops.timeSeries("Trig", 2, t0, t0 + cycNum * period, period)
    ops.pattern("Plain", 2, 2)
    for nd_ in (5, 6, 7, 8):
        ops.load(nd_, 0, 0, q_amp / 4.0, 0)

    # run + collecte
    t_end = t0 + cycNum * period + 1e-12
    t_hist=[]; u_hist=[]
    p_hist=[]; q_hist=[]
    dT_cur=dT

    # stress/strain init pour référence (comme dans mono)
    s0 = np.asarray(ops.eleResponse(1, "stress"), float)
    e0 = np.asarray(ops.eleResponse(1, "strain"), float)
    if s0.size < 6 or e0.size < 6:
        ops.wipe()
        return 1, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    SIGX0 = -s0[0]
    SIGZ0 = -s0[2]
    SIG0  = SIGX0 - abs(pConf)  # même offset que ton mono

    while ops.getTime() < t_end:
        ok = ops.analyze(1, dT_cur)
        if ok != 0:
            tries=0
            while ok != 0 and dT_cur>dTmin and tries<100:
                dT_cur*=0.5
                ok=ops.analyze(1,dT_cur); tries+=1
            if ok != 0:
                break

        t_hist.append(ops.getTime())
        u_hist.append(ops.nodeVel(8, 4))  # Us

        s = np.asarray(ops.eleResponse(1, "stress"), float)
        e = np.asarray(ops.eleResponse(1, "strain"), float)
        if s.size < 6 or e.size < 6:
            break

        # contraintes effectives (sign convention identique à ton mono)
        SIGX = -s[0] - SIG0

        SIGZ = -s[2] - SIG0

        p = (SIGX*2 + SIGZ) / 3.0
        q = SIGZ - SIGX  # comme dans mono


        p_hist.append(p)
        q_hist.append(q)


    ops.wipe()

    if len(t_hist) == 0:
        return 1, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    t_abs = np.asarray(t_hist,float)
    t_rel = t_abs - t0
    mask  = t_rel >= 0.0

    Ns = t_rel[mask]/period
    Us = np.asarray(u_hist,float)[mask]
    ps = np.asarray(p_hist,float)[mask]
    qs = np.asarray(q_hist,float)[mask]


    return ok, Ns, Us, ps, qs

