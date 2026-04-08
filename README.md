# Configuration-aware-natural-gradient-vqc
Code and results for Configuration-Aware Natural Gradient Optimization for Variational Quantum Circuits
# Configuration-Aware Natural Gradient Optimization for Variational Quantum Circuits

This repository provides the implementation and reproducible simulation framework accompanying the study on configuration-aware optimization in variational quantum circuits.

## Reproducibility Statement

The results reported in the manuscript are generated using a controlled and deterministic simulation pipeline implemented in Python. The study investigates optimization under configuration variability, where each admissible circuit realization induces a distinct objective over a shared parameter space.

The implementation includes:

- exact statevector simulation,
- parameter-shift gradient computation,
- evaluation of the quantum geometric tensor,
- explicit construction of configuration families,
- fixed random seeds to ensure reproducibility.

## Execution

The main experiment is implemented in:

favo_final_experiment.py

To reproduce the reported results, execute:

python favo_final_experiment.py --seeds 0 1 2 3 4 5 6 7 8 9 --variability-levels 0.00 0.01 0.02 0.05

## Outputs

The execution produces:

- run-level experiment data,
- aggregated performance summaries,
- paired method comparisons,
- variability sweep statistics,
- figures corresponding to the reported results.

All outputs are deterministically generated under fixed seeds and configuration parameters.
