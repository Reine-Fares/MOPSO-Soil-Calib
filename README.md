# Soil Model Calibration with Adaptive Multi-Objective PSO

This repository provides a Python framework for the automated calibration of soil constitutive models using an adaptive α-weighted multi-objective Particle Swarm Optimization (PSO) approach.

The calibration procedure combines cyclic and monotonic triaxial test simulations with a multi-objective loss function to identify optimal constitutive parameters.

Although the example implementation provided here uses the Manzari–Dafalias constitutive model, the workflow can be adapted to other soil constitutive models by modifying the simulation routines. The same can be said to the OpenSees code. 

# Repository structure

The main scripts required to run the calibration are:

- main.py – main script that launches the calibration workflow
- pso_calibration.py – PSO optimization engine and particle evaluation routines
- cyclic_triaxialtest.py – cyclic triaxial simulation using OpenSees
- monotonic_triaxialtest.py – monotonic triaxial simulation using OpenSees
- cost_functions.py – cyclic and monotonic cost functions used during calibration

- mainparameters.py – global configuration parameters including plotting option and loss funtion parameter
- soilparameters.py – soil constitutive model parameters, parameter to calibrate + calibration bounds and steps and solver parameters 
- psoparameters.py – parameters controlling the PSO algorithm  
- exp_data.py – loading and preprocessing of experimental data  


# How to run the calibration

The calibration procedure is launched using:
python main.py


# Parameter configuration

Most user-adjustable settings are located in files containing **`parameters`** in their name:

mainparameters.py
soilparameters.py
psoparameters.py

Some parameters are marked with the symbol:
!!  = parameter value recommended by the authors (should not be modified)
These values correspond to the configuration used in the reference study.
Reference : Multi-Objective Particle Swarm Optimization Calibration of Saturated Soil Constitutive Models

# Experimental data

The framework requires experimental datasets for:
- cyclic triaxial tests (N,u)
- monotonic triaxial tests (eps-q)
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
However, the framework can be adapted to other soil constitutive models by modifying : 
- cyclic_triaxialtest.py
- monotonic_triaxialtest.py
their respective call in psoparameters.py, and 
- soilparameter.py
  
# Output

Depending on the configuration, the code produces:

- calibrated cyclic response curves  
- calibrated monotonic response curves  
- CSV files containing optimization results  
- plots showing error evolution with α  
- optimized parameter values  

# General References
OpenSees resources:
Zhu, M., McKenna, F., & Scott, M. H. (2018). OpenSeesPy: Python library for the OpenSees finite element framework. SoftwareX, 7, 6-11.
https://doi.org/10.1016/j.softx.2017.10.009

OpenSees documentation:  
https://opensees.github.io/OpenSeesDocumentation


# How to Cite
If you used MOPSO-Soil-Calib for a scientific paper, please cite it as:
Fares, R., Uzquiano Al-Ricabi, F., Lopez Caballero, F., 2026. MOPSO-Soil-Calib – Calibration of a Multi-Objective Loss Function Applied to Saturated Soil Constitutive Models. [https://doi.org/10.5281/zenodo.1904811]

This repository accompanies the following research work:


