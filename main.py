"""
Multi-Objective Particle Swarm Optimization Calibration of Saturated Soil Constitutive Models
Reine Fares, Fernando Uzquiano Al-Ricabi, Fernando Lopez Caballero
=========================================

This script evaluates multiple combinations of cyclic and monotonic cost
functions for model calibration using Particle Swarm Optimization (PSO).

Depending on the configuration, the script can either perform:

    • a direct calibration using a single alpha value, or
    • an alpha scan if an array of alpha values is provided.

For each cost-function pair, the script:
    1. Performs a direct calibration if a single alpha is defined,
       or scans a grid of alpha values if an array is specified.
    2. Runs PSO optimization
    3. Saves calibrated curves
    4. Stores summary statistics
    5. Generates error bar plots

Cost Function Sets
------------------

Cyclic cost functions:
    - N70
    - N80
    - N90
    - avg3
    - rmse

Monotonic cost functions:
    - nrmse
    - yield
    - qmax


Outputs
-------

For each (mono_cost_name, cyc_cost_name) combination:
    - Alpha scan CSV results
    - Calibrated curve CSV exports
    - Error bar plots
    - Summary table of optimization metrics

Glossary
--------
# !! = value recommended by the authors — do not modify

Authors
------
Reine FARES
Université Paris-Saclay, CEA, Service d’Études Mécaniques et Thermiques, 91191 Gif-sur-Yvette, France

Fernando Uzquiano Al-Ricabi
Université Paris-Saclay, CEA, Service d’Études Mécaniques et Thermiques, 91191 Gif-sur-Yvette, France

Fernando Lopez Caballero
LMPS – Laboratoire de Mécanique Paris-Saclay, Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, 91190 Gif-Sur-Yvette, France
"""


# =============================================================================
# Imports
# =============================================================================
import os, math, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

import exp_data as EXP

import mainparameters as MPA
import soilparameters as SPA
import psoparameters as PPA

import pso_calibration as PSC
 




# ===========================
# Main
# ===========================
def main():
    MAX_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print("MAX_WORKERS =", MAX_WORKERS)

    cycNum = int(max(SPA.CYCNUM, round(float(np.nanmax(EXP.N_exp)))))
    print("cycNum =", cycNum)


    for cyc_cost_name, mono_cost_name in itertools.product(
        MPA.CYCLIC_COSTS, MPA.MONOTONIC_COSTS
    ):
        outdir = MPA.OUTROOT / f"{mono_cost_name}_{cyc_cost_name}"
        outdir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n========== Combo: MONO={mono_cost_name} | CYC={cyc_cost_name} =========="
        )

        results = []
        cache_best = dict(J=np.inf, alpha=None, pars=None)

        for alpha_raw in MPA.ALPHAS:
            alpha = float(np.clip(alpha_raw, -1.0, 1.0))
            w_cyc = 0.5 * (1.0 + alpha)
            w_mono = 0.5 * (1.0 - alpha)
            print(f"\n=== α={alpha:+.2f} → w_cyc={w_cyc:.2f}, w_mono={w_mono:.2f} ===")

            cyc_cfg = dict(period=SPA.period, cycNum=cycNum, CSR=SPA.CSR, w=w_cyc)
            mono_cfg = dict(devDisp=SPA.DEVDISP, w=w_mono)
            
            # -----------------------------------------------------------------
            # Run PSO optimization 
            # -----------------------------------------------------------------

            gpos, gbest_cost, art = PSC.pso_parallel(
                SPA.materiau_base,
                SPA.BOUNDS,
                cyc_cfg,
                mono_cfg,
                cyc_cost_name,
                mono_cost_name,
                n_particles=PPA.N_PART,
                n_iters=PPA.N_ITERS,
                seed=PPA.SEED,
                vfrac=PPA.VFRAC,
                max_workers=MAX_WORKERS,
            )


            J_cyc_rep = np.nan
            J_mono_rep = np.nan
            if art["pars"] is not None:

                
                mmc = art.get("mmc", None)
                sc  = art.get("sc", None)
                mm  = art.get("mm", None)
                sm  = art.get("sm", None)
                
                J_tot_rep, J_cyc_rep, J_mono_rep = PSC.eval_J_only(
                    art["pars"], cyc_cfg, mono_cfg, cyc_cost_name, mono_cost_name,
                    mmc=mmc, sc=sc, mm=mm, sm=sm
                )
                
                
            print(
                f" → α={alpha:+.2f}: J_cyc={J_cyc_rep:.4g}, J_mono={J_mono_rep:.4g}, J_tot={J_tot_rep:.4g}"
            )

            # -----------------------------------------------------------------
            # Data export
            # -----------------------------------------------------------------

            if art["N"] is not None and art["u"] is not None and np.size(art["N"]) > 0:
                pd.DataFrame({"N": art["N"], "u_kPa": art["u"], "p_kPa": art["psc"], "q_kPa": art["qsc"]}).to_csv(
                    outdir / f"calib_cyclic_{cyc_cost_name}_alpha_{alpha:+.2f}.csv",
                    index=False,
                )
            if (
                art["eps"] is not None
                and art["q"] is not None
                and np.size(art["eps"]) > 0
            ):
                pd.DataFrame({"eps1_pct": art["eps"], "q_kPa": art["q"]}).to_csv(
                    outdir / f"calib_monotonic_{mono_cost_name}_alpha_{alpha:+.2f}.csv",
                    index=False,
                )

            # ligne résultats (inclut gpos)
            row = dict(
                alpha=float(alpha),
                w_cyc=w_cyc,
                w_mono=w_mono,
                cyc_cost=cyc_cost_name,
                mono_cost=mono_cost_name,
                J_cyc=float(J_cyc_rep),
                J_mono=float(J_mono_rep),
                J_total=float(J_tot_rep),
            )
            for i, k in enumerate(SPA.PARAM_NAMES):
                row[k] = float(gpos[i])
            results.append(row)

            # garder meilleur alpha*
            if np.isfinite(J_tot_rep) and J_tot_rep < cache_best["J"]:
                cache_best.update(J=J_tot_rep, alpha=alpha, pars=art["pars"])

        # synthèse combo
        df = pd.DataFrame(results).sort_values("J_total").reset_index(drop=True)
        out_csv = outdir / f"alpha_scan_{mono_cost_name}_{cyc_cost_name}.csv"
        df.to_csv(out_csv, index=False)
        best = df.iloc[0]
        print(
            f"\n→ COMBO FINI: MONO={mono_cost_name} | CYC={cyc_cost_name} | "
            f"α*={best['alpha']:+.2f} (J_tot={best['J_total']:.4f}) | CSV: {out_csv}"
        )

        # -----------------------------------------------------------------
        # Plotting
        # -----------------------------------------------------------------


        if MPA.ploting == 0 :
            dfp = df.sort_values("alpha")
            x = np.arange(len(dfp))
            bw = 0.28
            plt.figure(figsize=(8, 5))
            plt.bar(x - bw, dfp["J_cyc"], width=bw, label=f"Cyclic ({cyc_cost_name})")
            plt.bar(x, dfp["J_mono"], width=bw, label=f"Monotonic ({mono_cost_name})")
            plt.bar(x + bw, dfp["J_total"], width=bw, label="Global J_tot")
            plt.xticks(x, [f"{a:+.2f}" for a in dfp["alpha"]])
            plt.xlabel("α")
            plt.ylabel("Error")
            plt.title(f"Error vs α — {mono_cost_name} + {cyc_cost_name}")
            plt.grid(axis="y", ls=":")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "alpha_errors_barplot.png", dpi=180)
            plt.close()

if __name__ == "__main__":
    main()
