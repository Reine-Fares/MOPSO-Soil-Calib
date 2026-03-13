# -*- coding: utf-8 -*-
"""
Alpha-scan HN31 — 15 combinaisons de fonctions coût (PSO parallèle) + Sensibilité OAT
- Cyclique: 'N70' | 'N80' | 'N90' | 'avg3' | 'rmse'
- Monotone: 'nrmse' | 'yield' | 'qmax'
- Poids: alpha_cyc = 0.5*(1+alpha), alpha_mono = 0.5*(1-alpha)  (PAS de renormalisation)
- Sensibilité (par combo): OAT sur {A0, cz, ch, z_max, h0} au meilleur alpha*, ±10%

Lancer:  python3 pso_alpha_multi_costs_all.py
"""

import os, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# -- réduire la sur-parallélisation BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import mainparameters

# ===========================
# Évaluation d'une particule
# ===========================

import alphaoptimization as AOP


# ===========================
# Main
# ===========================
def main():
    MAX_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print("MAX_WORKERS =", MAX_WORKERS)

    cycNum = int(max(CYCNUM, round(float(np.nanmax(N_exp)))))
    print("cycNum =", cycNum)


    for cyc_cost_name, mono_cost_name in itertools.product(
        CYCLIC_COSTS, MONOTONIC_COSTS
    ):
        outdir = OUTROOT / f"{mono_cost_name}_{cyc_cost_name}"
        outdir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n========== Combo: MONO={mono_cost_name} | CYC={cyc_cost_name} =========="
        )

        results = []
        cache_best = dict(J=np.inf, alpha=None, pars=None)

        for alpha_raw in ALPHAS:
            alpha = float(np.clip(alpha_raw, -1.0, 1.0))
            # Poids EXPLICITES (doc) — PAS de renormalisation:
            w_cyc = 0.5 * (1.0 + alpha)
            w_mono = 0.5 * (1.0 - alpha)
            print(f"\n=== α={alpha:+.2f} → w_cyc={w_cyc:.2f}, w_mono={w_mono:.2f} ===")

            cyc_cfg = dict(period=period, cycNum=cycNum, CSR=CSR, w=w_cyc)
            mono_cfg = dict(devDisp=DEVDISP, w=w_mono)


            # consider smaller vfrac (0.25 → 0.15):
            # vmax = (up - low) * 0.15
            # This alone often prevents “cz=400, zmax=10” attractor.

            # PSO
            gpos, gbest_cost, art = AOP.pso_parallel(
                materiau_base,
                BOUNDS,
                cyc_cfg,
                mono_cfg,
                cyc_cost_name,
                mono_cost_name,
                n_particles=N_PART,
                n_iters=N_ITERS,
                seed=SEED,
                vfrac=VFRAC,
                max_workers=MAX_WORKERS,
            )

            # Reporting local
            # recalcul propre avec les poids courants (déjà dans J de l'artefact via eval_particle)
            J_cyc_rep = np.nan
            J_mono_rep = np.nan
            if art["pars"] is not None:

                
                mmc = art.get("mmc", None)
                sc  = art.get("sc", None)
                mm  = art.get("mm", None)
                sm  = art.get("sm", None)
                
                J_tot_rep, J_cyc_rep, J_mono_rep = AOP.eval_J_only(
                    art["pars"], cyc_cfg, mono_cfg, cyc_cost_name, mono_cost_name,
                    mmc=mmc, sc=sc, mm=mm, sm=sm
                )
                
                
            print(
                f" → α={alpha:+.2f}: J_cyc={J_cyc_rep:.4g}, J_mono={J_mono_rep:.4g}, J_tot={J_tot_rep:.4g}"
            )

            # figures & CSV courbes
            # Courbes CSV
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
            for i, k in enumerate(PARAM_NAMES):
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

        # barplot erreurs vs alpha
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
