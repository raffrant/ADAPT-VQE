# Quantum VQE and Machine Learning

A research repository at the intersection of **variational quantum algorithms**, **measurement-based quantum-state engineering**, and **classical machine learning for protocol discovery**. The current codebase naturally separates into two main directions: an **ADAPT-VQE** track for compact ansatz construction in quantum chemistry, and a **machine-learning / graph-state** track for predicting high-quality measurement protocols on weighted graph states. 

## Repository overview

This repository contains two complementary lines of work. The first focuses on **ADAPT-VQE** for molecular ground-state preparation, where the ansatz is built adaptively by selecting the operator with the largest energy gradient at each iteration rather than fixing the circuit in advance. The second studies **weighted graph states under local measurements**, using sector lengths, cluster fidelity, and supervised learning to identify measurement protocols that concentrate useful multipartite entanglement. 

## Folder structure

```text
Quantum_vqe_and_ml/
├── adapt_vqe/
│   ├── h4closed.py
│   ├── h4closedfigure.jpg
│   └── ...
├── mlandgraphstates/
│   ├── mlandsectorlengths.py
│   ├── sectorlengths.py
│   ├── graph_state_notebook.ipynb
│   └── ...
└── README.md
```

The `adapt_vqe/` folder is centered on an H4 closed-shell ADAPT-VQE implementation in JAX/SciPy, while the `machine_learning/` folder contains weighted graph-state simulation, sector-length analysis, dataset generation, and ML-based ranking of measurement protocols. 

## ADAPT-VQE

The ADAPT-VQE part of the repository implements an adaptive variational eigensolver for the H4 molecule after Jordan–Wigner transformation, using JAX for fast differentiable linear algebra and SciPy BFGS for parameter optimization. The uploaded code and README describe a gradient-driven operator-selection loop, Hartree–Fock initialization, an operator pool built from particle-number/symmetry-respecting Pauli strings, and convergence tracking in both **energy error** and **wavefunction infidelity**. 

The included convergence figure shows that the H4 workflow reaches near-exact agreement in both energy and wavefunction after a small number of adaptive iterations, which is exactly the kind of compact-ansatz behavior ADAPT-VQE is designed to exploit. In the current implementation, `h4closed.py` is the main driver script, and it expects access to precomputed Hamiltonian data for the H4 problem. 

## Machine learning on graph states

The machine-learning side of the repository studies weighted graph states on small lattices, applies local projective measurements to selected qubits, and evaluates the surviving subsystem using both **cluster-state fidelity** and **sector lengths** as structural descriptors. The main Python pipeline in `mlandsectorlengths.py` generates protocol-level datasets, trains regression and classification models, scores candidate protocols, and ranks the most promising measurement patterns automatically. 

The notebook `graph_state_notebook.ipynb` is the exploratory front end for this workflow: it generates datasets, inspects fidelity and sector-length distributions, evaluates ML metrics, and visualizes the best raw and ML-ranked protocols. The uploaded notebook output shows a representative dataset with 6000 protocol rows and reports example model metrics including sector-length regression, fidelity regression, and good-output classification. 

## Main files

### `adapt_vqe/`
- `h4closed.py` — main ADAPT-VQE loop for the H4 closed-shell problem, using JAX, custom operator pools, and BFGS optimization. 
- `h4closedfigure.jpg` — convergence figure for energy and wavefunction infidelity. 

### `mlandgraphstates/`
- `mlandsectorlengths.py` — end-to-end pipeline for weighted graph states, measurement protocols, sector lengths, and ML ranking. 
- `sectorlengths.py` — exploratory script for sector-length calculations and related state analysis. 
- `graph_state_notebook.ipynb` — notebook for dataset generation, visualization, and model evaluation. 

## Installation

The exact environment differs slightly by folder, but the uploaded files indicate a Python workflow built around scientific computing, plotting, and machine learning packages. A practical starting point is: 

```bash
pip install numpy scipy matplotlib pandas seaborn scikit-learn joblib tqdm jupyter jax jaxlib
```

For the ADAPT-VQE code, JAX 64-bit mode is explicitly enabled in the script, which is important for numerical stability in chemistry-style calculations. The graph-state ML pipeline additionally depends on pandas, seaborn, scikit-learn, joblib, and tqdm for dataset generation, visualization, and model training. 

## How to use

### Run ADAPT-VQE

1. Place the precomputed H4 Hamiltonian data file in the location expected by `h4closed.py`. 
2. Adjust the output paths used to save selected operators and optimized parameters. 
3. Run:

```bash
python adapt_vqe/h4closed.py
```

This workflow performs adaptive operator selection, re-optimizes all variational parameters after each addition, and tracks convergence in both energy error and wavefunction infidelity. 

### Run the graph-state ML pipeline

You can either run the notebook interactively or execute the Python pipeline directly. The notebook is useful for inspection and plotting, while `mlandsectorlengths.py` is the reproducible script-style entry point. 

```bash
python mlandgraphstates/mlandsectorlengths.py
```

Typical outputs include protocol datasets, regression/classification metrics, scored protocols, best-per-graph selections, and summary figures saved to an `output/` directory. 

## Current status

This repository is best viewed as a **research prototype collection** rather than a polished library. The files already contain substantial scientific logic, but they would benefit from cleaner packaging, path/config management, reusable modules, and more standardized experiment outputs. 

## Future directions

Natural next steps for the repository include: 

- refactoring both branches into importable modules,
- adding environment files and reproducible configs,
- saving figures and metrics in a more uniform structure,
- adding a short methods note for each folder,
- extending the ML branch to larger lattices or alternative target states,
- and extending the ADAPT-VQE branch to more molecules, operator pools, or hardware-efficient comparisons.

## Notes

If this repository is public-facing, a good presentation strategy is to keep this top-level README focused on the big picture and then give each folder its own detailed README. That way the root page explains the scientific identity of the repository, while `adapt_vqe/` and `machine_learning/` can contain workflow-specific setup and usage notes. 
