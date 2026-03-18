"""
PSO Calibration Engine
======================

This module implements the Particle Swarm Optimization (PSO) used for the calibration process

It provides:

    • cost evaluation for cyclic and monotonic tests
    • normalization of cost components with a warm-up SPA.STEP
    • particle evaluation routines
    • parallel PSO optimization
    • restart strategy for stagnation

"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
from concurrent.futures import ProcessPoolExecutor

import exp_data as EXP 
import soilparameters as SPA
import mainparameters as MPA
import psoparameters as PPA

import cost_functions as COF
import cyclic_triaxialtest as CTT
import monotonic_triaxialtest as MTT


# =============================================================================
# Combines normalized cyclic and monotonic costs
# =============================================================================

def compute_global_J(J_cyc_raw, J_mono, J_cov_pen, w_cyc, w_mono, mmc, sc, mm, sm, eps=1e-12):
    Jc_u = (J_cyc_raw - mmc) / (sc + eps)
    Jm_u = (J_mono    - mm ) / (sm + eps)
    J_lin = w_cyc * (Jc_u + J_cov_pen) + w_mono * Jm_u
    J_tot = MPA.BETA1 * J_lin + MPA.BETA2 * abs(Jc_u  - Jm_u) 
    return float(J_tot), float(Jc_u), float(Jm_u)

# =============================================================================
# Evaluates a particle during PSO optimization
# =============================================================================

def eval_raw_costs(vec, base_params, cyc_cfg, mono_cfg, cyc_cost_name, mono_cost_name):
    vec = COF.snap_to_grid(np.asarray(vec, float), SPA.LOW, SPA.UP, SPA.STEP)
    pars = dict(base_params)
    for k, v in zip(SPA.PARAM_NAMES, vec):
        pars[k] = float(v)

    
    status, Ns, Us, psc, qsc  = CTT.run_cyclic_triaxial(
    pars,
    period=cyc_cfg["period"],
    cycNum=cyc_cfg["cycNum"],
    Tcc=cyc_cfg["CSR"],
    dT=SPA.dT,
    dTmin=SPA.dTmin,
    )

    # --- raw cyclic cost (NO coverage penalty here) ---
    if status == 0 and Ns.size > 1:
        cc = cyc_cost_name.lower()
        if cc == "rmse":
            J_cyc = COF.cost_cyc_rmse(Ns, Us, EXP.N_exp, EXP.u_exp)
        elif cc == "juu":
            J_cyc = COF.cost_u_rmse_u(Ns, Us, EXP.N_exp, EXP.u_exp) 
        elif cc == "avg3":
            J_cyc = COF.cost_cyc_avg3(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"])
        elif cc == "uN80":
            J_cyc = COF.cost_N80(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"])           
        elif cc in ("n70", "n80", "n90"):
            th = {"n70": 0.7, "n80": 0.8, "n90": 0.9}[cc]
            J_cyc = COF.cost_cyc_Ntheta(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"], theta=th)
        else : 
            J_cyc = np.inf

    else:
        J_cyc = np.inf

    # coverage
    N_target = float(cyc_cfg["cycNum"])
    N_reached = float(np.max(Ns)) if (status == 0 and Ns.size) else 0.0
    cov = N_reached / max(N_target, 1e-12)

    # --- raw monotonic cost ---

    status, epsS, qS = MTT.run_monotonic_triaxial(pars, devDisp=mono_cfg["devDisp"])
    mc = mono_cost_name.lower()
    if status == 0 and qS.size > 1:
        if mc == "nrmse":
            J_mono = COF.cost_mono_qeps(epsS, qS, EXP.eps_exp, EXP.q_exp)
        elif mc == "yield":
            J_mono = COF.cost_mono_yield(epsS, qS, EXP.eps_exp, EXP.q_exp)
        elif mc == "qmax":
            J_mono = COF.cost_mono_qmax(epsS, qS)
        else:
            J_mono = np.inf

    else :
        J_mono = np.inf

    return float(J_cyc), float(J_mono), float(cov)

# =============================================================================
# Computes cyclic and monotonic costs
# =============================================================================

def eval_particle(vec, base_params, cyc_cfg, mono_cfg, cyc_cost_name, mono_cost_name, mmc=None , sc=None ,mm=None, sm=None):
    vec = COF.snap_to_grid(np.asarray(vec, float), SPA.LOW, SPA.UP, SPA.STEP)
    pars = dict(base_params)
    for k, v in zip(SPA.PARAM_NAMES, np.asarray(vec, float)):
        pars[k] = float(v)

    status, Ns, Us, psc, qsc  = CTT.run_cyclic_triaxial(
    pars,
    period=cyc_cfg["period"],
    cycNum=cyc_cfg["cycNum"],
    Tcc=cyc_cfg["CSR"],
    dT=SPA.dT,
    dTmin=SPA.dTmin,
    )
    
    if status == 0 and Ns.size > 1:
        cc = cyc_cost_name.lower()
        if cc == "rmse":
            J_cyc_raw  = COF.cost_cyc_rmse(Ns, Us, EXP.N_exp, EXP.u_exp)
            J_cov_pen = 0.0
        elif cc == "juu":
            J_cyc = COF.cost_u_rmse_u(Ns, Us, EXP.N_exp, EXP.u_exp) 
            J_cov_pen = 0.0
        elif cc == "avg3":
            J_cyc = COF.cost_cyc_avg3(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"])
            J_cov_pen = 0.0
        elif cc == "uN80":
            J_cyc = COF.cost_N80(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"])
            J_cov_pen = 0.0            
        elif cc in ("n70", "n80", "n90"):
            th = {"n70": 0.7, "n80": 0.8, "n90": 0.9}[cc]
            J_cyc = COF.cost_cyc_Ntheta(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"], theta=th)
            J_cov_pen = 0.0
        else : 
            J_cov_pen = np.inf
            J_cyc_raw  = np.inf
            
        N_target = float(cyc_cfg["cycNum"])   # 17
        N_reached = float(np.max(Ns)) if Ns.size else 0.0
        cov = N_reached / max(N_target, 1e-12)

        if EXP.LIQ == 0 : 
            Ncyc80 = COF.Nref_N80(EXP.N_exp, EXP.u_exp, pars["pConf"], 0.8)
        else :
            u_scalar = float(EXP.u_exp.max())
            bounded_u = min(u_scalar , pars["pConf"])
            Ncyc80 = Nref_N80(EXP.N_exp, EXP.u_exp, bounded_u , 0.8)
        covref = Ncyc80 / max(N_target, 1e-12) * 1.1

        if cov < covref:
            J_cov_pen = 50.0 * ((covref - cov) / covref) ** 2
    else:
        J_cov_pen = np.inf
        J_cyc_raw = np.inf
        

    status, epsS, qS = MTT.run_monotonic_triaxial(pars, devDisp=mono_cfg["devDisp"])
    if status == 0 and qS.size > 1:
        mc = mono_cost_name.lower()
        if mc == "nrmse":
            J_mono = COF.cost_mono_qeps(epsS, qS, EXP.eps_exp, EXP.q_exp)
        elif mc == "yield":
            J_mono = COF.cost_mono_yield(epsS, qS, EXP.eps_exp, EXP.q_exp)
        elif mc == "qmax":
            J_mono = COF.cost_mono_qmax(epsS, qS)
            
        else : 
            J_mono = np.inf
    else: J_mono =np.inf
        

    w_cyc = float(cyc_cfg["w"])
    w_mono = float(mono_cfg["w"])

    J, Jc_used, Jm_used = compute_global_J(J_cyc_raw, J_mono, J_cov_pen, w_cyc, w_mono, mmc , sc ,mm , sm )
    
    if not np.isfinite(J):
        J = np.inf
    return float(J), float(Jc_used), float(Jm_used), pars, (Ns, Us), (epsS, qS) ,(psc,qsc)

# =============================================================================
# Compute cyclic and monotonic costs 
# =============================================================================

def eval_J_only(pars, cyc_cfg, mono_cfg, cyc_cost_name, mono_cost_name, mmc=None , sc=None ,mm=None, sm=None ):

    status, Ns, Us, psc, qsc  = CTT.run_cyclic_triaxial(
    pars,
    period=cyc_cfg["period"],
    cycNum=cyc_cfg["cycNum"],
    Tcc=cyc_cfg["CSR"],
    dT=SPA.dT,
    dTmin=SPA.dTmin,
    )
    
    if status == 0 and Ns.size > 1:
        cc = cyc_cost_name.lower()
        if cc == "rmse":
            J_cyc_raw  = COF.cost_cyc_rmse(Ns, Us, EXP.N_exp, EXP.u_exp)
            J_cov_pen = 0.0
        elif cc == "juu":
            J_cyc = COF.cost_u_rmse_u(Ns, Us, EXP.N_exp, EXP.u_exp) 
            J_cov_pen = 0.0
        elif cc == "avg3":
            J_cyc = COF.cost_cyc_avg3(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"])
            J_cov_pen = 0.0
        elif cc == "uN80":
            J_cyc = COF.cost_N80(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"])
            J_cov_pen = 0.0            
        elif cc in ("n70", "n80", "n90"):
            th = {"n70": 0.7, "n80": 0.8, "n90": 0.9}[cc]
            J_cyc = COF.cost_cyc_Ntheta(Ns, Us, EXP.N_exp, EXP.u_exp, pars["pConf"], theta=th)
            J_cov_pen = 0.0            
 
        else : 
            J_cov_pen = np.inf
            J_cyc_raw  = np.inf
            
        N_target = float(cyc_cfg["cycNum"])   # 17
        N_reached = float(np.max(Ns)) if Ns.size else 0.0
        cov = N_reached / max(N_target, 1e-12)
        
        if EXP.LIQ == 0 : 
            Ncyc80 = COF.Nref_N80(EXP.N_exp, EXP.u_exp, pars["pConf"], 0.8)
        else :
            u_scalar = float(EXP.u_exp.max())
            bounded_u = min(u_scalar , pars["pConf"])
            Ncyc80 = Nref_N80(EXP.N_exp, EXP.u_exp, bounded_u , 0.8)
            
        covref = Ncyc80 / max(N_target, 1e-12) * 1.1

        if cov < covref:
            J_cov_pen = 50.0 * ((covref - cov) / covref) ** 2
    else:
        J_cov_pen = np.inf 
        J_cyc_raw = np.inf
        

    status, epsS, qS = MTT.run_monotonic_triaxial(pars, devDisp=mono_cfg["devDisp"])
    if status == 0 and qS.size > 1:
        mc = mono_cost_name.lower()
        if mc == "nrmse":
            J_mono = COF.cost_mono_qeps(epsS, qS, EXP.eps_exp, EXP.q_exp)
        elif mc == "yield":
            J_mono = COF.cost_mono_yield(epsS, qS, EXP.eps_exp, EXP.q_exp)
        elif mc == "qmax":
            J_mono = COF.cost_mono_qmax(epsS, qS)    
        else : 
            J_mono = np.inf
    else: J_mono =np.inf
        

        
    w_cyc = float(cyc_cfg["w"])
    w_mono = float(mono_cfg["w"])

    J, Jc_used, Jm_used = compute_global_J(J_cyc_raw, J_mono, J_cov_pen, w_cyc, w_mono, mmc , sc ,mm , sm )
    return float(J), float(Jc_used), float(Jm_used)


# =============================================================================
# Parallel PSO optimizer used for parameter calibration
# =============================================================================


def pso_parallel(
    base_params,
    bounds,
    cyc_cfg,
    mono_cfg,
    cyc_cost_name,
    mono_cost_name,
    n_particles=100,
    n_iters=30,
    seed=123,
    vfrac=0.15,
    max_workers=None,
):
    rng = np.random.default_rng(seed)
    low, up = np.asarray(bounds[0]), np.asarray(bounds[1])
    d = len(low)
    pos = rng.uniform(low, up, size=(n_particles, d))
    pos = COF.snap_to_grid(pos, low, up, SPA.STEP)

    # --- warm-up refs 
    # PPA.n_ref = 100 # increase to stabilize ref estimates
    ref_pos = rng.uniform(low, up, size=(PPA.n_ref, d))
    ref_pos = COF.snap_to_grid(ref_pos, low, up, SPA.STEP)
    
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                eval_raw_costs,
                ref_pos[i],
                base_params,
                cyc_cfg,
                mono_cfg,
                cyc_cost_name,
                mono_cost_name,
            )
            for i in range(PPA.n_ref)
        ]
        res0 = [f.result() for f in futs]
    
    Jc0  = np.array([r[0] for r in res0], float)
    Jm0  = np.array([r[1] for r in res0], float)
    cov0 = np.array([r[2] for r in res0], float)
    

    
    N_target = float(cyc_cfg["cycNum"])
    pars = dict(base_params)
    if EXP.LIQ == 0 : 
        Ncyc80 = COF.Nref_N80(EXP.N_exp, EXP.u_exp, pars["pConf"], 0.8)
    else :
        u_scalar = float(EXP.u_exp.max())
        bounded_u = min(u_scalar , pars["pConf"])
        Ncyc80 = Nref_N80(EXP.N_exp, EXP.u_exp, bounded_u , 0.8)
    covref = Ncyc80 / max(N_target, 1e-12) * 1.1
    
    good_cyc  = np.isfinite(Jc0) & (Jc0 < 1e6) & (cov0 >= covref)
    good_mono = np.isfinite(Jm0) & (Jm0 < 1e6)
    print(f"[warmup] good_cyc={good_cyc.sum()}/{PPA.n_ref}, good_mono={good_mono.sum()}/{PPA.n_ref}")
    
    def robust_center_scale(x, min_n=10):
        x = np.asarray(x, float)
        if x.size < min_n:
            med = float(np.median(x)) if x.size else 1.0
            sc  = float(np.std(x)) if x.size else 1.0
            return med, max(sc, 1e-12)
        p25, p50, p75 = np.percentile(x, [25, 50, 75])
        sc = max(float(p75 - p25), 1e-12)
        return float(p50), sc
    
    Jc_ref = Jc0[good_cyc]
    Jm_ref = Jm0[good_mono]
    
    mmc, sc = robust_center_scale(Jc_ref, min_n=10)
    mm, sm = robust_center_scale(Jm_ref, min_n=10)
    
    print(f"[norm] cyc: med={mmc:.6g}, IQR={sc:.6g} | mono: med={mm:.6g}, IQR={sm:.6g}")    
    
    # (optional but useful for debug)
    # print(
        # "[warmup] cov stats:",
        # f"min={float(np.min(cov0)):.2f}",
        # f"p25={float(np.percentile(cov0,25)):.2f}",
        # f"med={float(np.median(cov0)):.2f}",
        # f"p75={float(np.percentile(cov0,75)):.2f}",
        # f"max={float(np.max(cov0)):.2f}",
    # )
    

    vmax = (up - low) * vfrac
    vel = rng.uniform(-vmax, vmax, size=(n_particles, d)) * 0.25
    pbest_pos = pos.copy()
    pbest_cost = np.full(n_particles, np.inf)
    gbest_pos = None
    gbest_cost = np.inf


    best_artifacts = dict(
    J=np.inf, pars=None, N=None, u=None, eps=None, q=None,
    mmc = mmc, sc = sc ,mm = mm, sm = sm
    )

    def schedule(t, T):

        T = max(T - 1, 1)
        w = PPA.w_max - (PPA.w_max - PPA.w_min) * t / T
        c1 = PPA.c1_max - (PPA.c1_max - PPA.c1_min) * t / T
        c2 = PPA.c2_min + (PPA.c2_max - PPA.c2_min) * t / T
        lower = 0.5 * (c1 + c2) - 1.0
        if w <= lower:
            w = lower + 1e-3
        if w >= 1.0:
            w = 0.999
        return w, c1, c2


    no_improve = 0
    best_so_far = np.inf
    span = (up - low)
    
    # --- exploration trackers ---
    min_seen = pos.min(axis=0).copy()
    max_seen = pos.max(axis=0).copy()


    for it in range(n_iters):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    eval_particle,
                    pos[i],
                    base_params,
                    cyc_cfg,
                    mono_cfg,
                    cyc_cost_name,
                    mono_cost_name,
                    # Jc_ref,
                    # Jm_ref,
                    mmc, sc, mm, sm, 
                )
                for i in range(n_particles)
            ]
            res = [f.result() for f in futs]
        costs = np.array([r[0] for r in res], float)
        
        
        # --- spread diagnostics (how much the swarm explores) ---
        min_seen = np.minimum(min_seen, pos.min(axis=0))
        max_seen = np.maximum(max_seen, pos.max(axis=0))
        
        # print only some iterations to avoid huge logs
        if it < 5 or (it + 1) % 5 == 0 or it == n_iters - 1:
            print("[spread] current swarm:")
            for j, name in enumerate(SPA.PARAM_NAMES):
                v = pos[:, j]
                print(
                    f"  {name}: min={v.min():.4g} p10={np.percentile(v,10):.4g} "
                    f"med={np.median(v):.4g} p90={np.percentile(v,90):.4g} max={v.max():.4g}"
                )
            print("[spread] ever visited (since init):")
            for j, name in enumerate(SPA.PARAM_NAMES):
                print(f"  {name}: min_seen={min_seen[j]:.4g} max_seen={max_seen[j]:.4g}")        
        
        
        
        improved = costs < pbest_cost
        pbest_cost[improved] = costs[improved]
        pbest_pos[improved] = pos[improved]
        i_min = int(np.argmin(pbest_cost))
        if gbest_pos is None or pbest_cost[i_min] < gbest_cost:
            gbest_cost = float(pbest_cost[i_min])
            gbest_pos = pbest_pos[i_min].copy()
            gbest_pos = COF.snap_to_grid(gbest_pos, low, up, SPA.STEP)
            
        if gbest_cost + 2e-4 < best_so_far:
            best_so_far = gbest_cost
            no_improve = 0
        else:
            no_improve += 1
            
        j_best_iter = int(np.argmin(costs))
        J, Jc, Jm, pars, (Ns, Us), (epsS, qS), (psc, qsc) = res[j_best_iter]
        if J < best_artifacts["J"]:
            best_artifacts.update(dict(J=J, pars=pars, N=Ns, u=Us, eps=epsS, q=qS, psc=psc, qsc=qsc))
        w, c1, c2 = schedule(it, n_iters)
        r1 = np.random.rand(n_particles, d)
        r2 = np.random.rand(n_particles, d)
        
        vel = PPA.chi* ( w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -vmax, vmax)

        prev_pos = pos.copy()
        pos = COF.snap_to_grid(prev_pos + vel, low, up, SPA.STEP)

        # optionnel mais fortement recommandé: recaler la vitesse sur le déplacement réellement appliqué
        vel = pos - prev_pos
        
        
        if no_improve >= PPA.STAG_ITERS:
            n_restart = int(max(1, round(PPA.RESTART_FRAC * n_particles)))
        
            order = np.argsort(pbest_cost)                 # ascending (best first)
            restart_idx = order[n_particles - n_restart :]
        
            # Split restarts: local around gbest + global uniform
            n_global = int(round(PPA.RESTART_GLOBAL_FRAC * n_restart))
            n_local = n_restart - n_global
        
            # Local: Gaussian around gbest
            center = gbest_pos.copy()
            noise = rng.normal(0.0, PPA.RESTART_SIGMA, size=(n_local, d)) * (up - low)
            new_local = COF.snap_to_grid(center + noise, low, up, SPA.STEP)
        
            # Global: Uniform over full bounds
            if n_global > 0:
                new_global = COF.snap_to_grid(rng.uniform(low, up, size=(n_global, d)), low, up, SPA.STEP)
                new_pos = np.vstack([new_local, new_global])
            else:
                new_pos = new_local
        
            # Apply restart
            pos[restart_idx] = new_pos
        
            # Reset velocity + pbest for restarted particles
            vel[restart_idx] = 0.0
            pbest_pos[restart_idx] = pos[restart_idx]
            pbest_cost[restart_idx] = np.inf
        
            print(f"[restart] stagnation={no_improve} -> restarted {n_restart}/{n_particles} "
                f"(local={n_local}, global={n_global})")
        
            no_improve = 0
                        
                
                
                
        
        gbest_pars = dict(zip(SPA.PARAM_NAMES, gbest_pos))
        print(f"  it {it+1:02d}/{n_iters} | gbest={gbest_cost:.6g} | no_improve={no_improve}")
        print(f"    gbest: h0={gbest_pars['h0']}, ch={gbest_pars['ch']}, A0={gbest_pars['A0']}, z_max={gbest_pars['z_max']}, cz={gbest_pars['cz']}")


    J, Jc, Jm, pars, (Ns, Us), (epsS, qS), (psc, qsc) = eval_particle(
        gbest_pos, base_params, cyc_cfg, mono_cfg, cyc_cost_name, mono_cost_name, mmc , sc ,mm , sm 
    )
    if J < best_artifacts["J"]:
        best_artifacts.update(dict(J=J, pars=pars, N=Ns, u=Us, eps=epsS, q=qS, psc=psc, qsc=qsc))
    gbest_pars = dict(zip(SPA.PARAM_NAMES, gbest_pos))
    print(f"  it {it + 1:02d}/{n_iters} | gbest={gbest_cost:.6g}, J_cyc={Jc:.4g}, J_mono={Jm:.4g}, J_tot={J:.4g}")
    print(f"    gbest: h0={gbest_pars['h0']}, ch={gbest_pars['ch']}, A0={gbest_pars['A0']}, z_max={gbest_pars['z_max']}, cz={gbest_pars['cz']}")
    return gbest_pos, gbest_cost, best_artifacts

