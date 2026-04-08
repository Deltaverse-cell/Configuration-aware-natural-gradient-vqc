# Configuration-aware-natural-gradient-vqc
Code and results for Configuration-Aware Natural Gradient Optimization for Variational Quantum Circuits
# Configuration-Aware Natural Gradient Optimization for Variational Quantum Circuits

Reproducibility Statement

This work presents a controlled simulation study of configuration-aware optimization in variational quantum circuits.

All results reported in the manuscript are generated using a deterministic simulation pipeline implemented in Python.

The implementation includes:
- exact statevector simulation,
- parameter-shift gradient computation,
- quantum geometric tensor evaluation,
- controlled configuration-family construction,
- fixed random seeds for reproducibility.

The main experiment is executed using the script:

favo_final_experiment.py

To reproduce the results, run:

python favo_final_experiment.py --seeds 0 1 2 3 4 5 6 7 8 9 --variability-levels 0.00 0.01 0.02 0.05

The outputs include:
- run-level results,
- aggregated summaries,
- paired comparisons,
- variability sweep statistics,
- figure generation outputs.

All simulations are deterministic under fixed seeds and configuration parameters.
