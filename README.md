# Soil Model Calibration with Adaptive Multi-Objective PSO

This repository provides a Python framework for the automated calibration of soil constitutive models using an adaptive α-weighted multi-objective Particle Swarm Optimization (PSO) approach.

The calibration procedure combines cyclic and monotonic triaxial test simulations with a multi-objective loss function to identify optimal constitutive parameters.

Although the example implementation provided here uses the Manzari–Dafalias constitutive model, the workflow can be adapted to other soil constitutive models by modifying the simulation routines. the same can be said to the OpenSees code. the mechanical code can substituted.



# Repository structure

The main scripts required to run the calibration are:


main.py
mainparameters.py
soilparameters.py
psoparameters.py
pso_calibration.py
cost_functions.py
exp_data.py
cyclic_triaxialtest.py
monotonic_triaxialtest.py


Description of the modules:

- `main.py` – main script that launches the calibration workflow  
- `mainparameters.py` – global configuration parameters and solver settings  
- `soilparameters.py` – soil constitutive model parameters and calibration bounds  
- `psoparameters.py` – parameters controlling the PSO algorithm  
- `pso_calibration.py` – PSO optimization engine and particle evaluation routines  
- `cost_functions.py` – cyclic and monotonic cost functions used during calibration  
- `exp_data.py` – loading and preprocessing of experimental data  
- `cyclic_triaxialtest.py` – cyclic triaxial simulation using OpenSees  
- `monotonic_triaxialtest.py` – monotonic triaxial simulation using OpenSees  



# How to run the calibration

The calibration procedure is launched using:
python main.py


The script performs the following operations:

1. Load experimental triaxial test data  
2. Run cyclic and monotonic simulations  
3. Evaluate model performance using cost functions  
4. Optimize model parameters using Particle Swarm Optimization  
5. Save calibrated responses and optimization results  


# Parameter configuration

Most user-adjustable settings are located in files containing **`parameters`** in their name:


mainparameters.py
soilparameters.py
psoparameters.py


These files define:

- solver configuration  
- PSO optimization parameters  
- calibration bounds  
- experimental loading conditions  
- loss function configuration  





Some parameters are marked with the symbol:
!!  = parameter value recommended by the authors (should not be modified)


These values correspond to the configuration used in the reference study.



# Experimental data

The framework requires experimental datasets for:
- cyclic triaxial tests
- monotonic triaxial tests
These datasets are loaded using the `exp_data.py` module.
The input data files must be placed in the appropriate directory specified in the configuration files.

# Optimization method

The calibration uses Particle Swarm Optimization (PSO) with several improvements:
- adaptive inertia and acceleration coefficients  
- restart strategy to avoid stagnation  
- parallel particle evaluation  
- α-weighted multi-objective loss function  



# Example constitutive model

The repository includes an example calibration of the Manzari–Dafalias sand constitutive model implemented in OpenSees.
However, the framework can be adapted to other soil constitutive models by modifying the simulation routines.


# Output

Depending on the configuration, the code produces:

- calibrated cyclic response curves  
- calibrated monotonic response curves  
- CSV files containing optimization results  
- plots showing error evolution with α  
- optimized parameter values  



# References

OpenSees resources:
Zhu, M., McKenna, F., & Scott, M. H. (2018). OpenSeesPy: Python library for the OpenSees finite element framework. SoftwareX, 7, 6-11.
https://doi.org/10.1016/j.softx.2017.10.009
OpenSees documentation:  
https://opensees.github.io/OpenSeesDocumentation

---


# How to Cite

This repository accompanies the following research work:

> *Automated calibration framework for soil constitutive models using an adaptive α-weighted multi-objective PSO approach.*

*(full citation here)*


