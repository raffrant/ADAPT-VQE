⚛️ ADAPT-VQE with JAX

A minimal implementation of the ADAPT-VQE algorithm for preparing the ground state of the hydrogen chain H₄ using JAX, https://github.com/google/jax
This project demonstrates how the Adaptive Variational Quantum Eigensolver (ADAPT-VQE) can be used to estimate molecular energies efficiently by iteratively constructing an ansatz.

✨ Features

🔹 Implementation of ADAPT-VQE for H₄
🔹 Uses JAX for automatic differentiation & linear algebra acceleration
🔹 Tracks energy difference estimation (ΔE) between iterations
🔹 Simple, modular, and easy to extend for other molecules/operators

📖 Background

The ADAPT-VQE algorithm improves upon traditional VQE by adaptively growing the variational ansatz:
Start with a reference state (Hartree–Fock).
Iteratively select operators based on their gradient contributions.
Optimize variational parameters until energy convergence.
Stop once the energy difference ΔE is below a chosen threshold.
This method often requires fewer parameters than fixed ansatz approaches, making it a promising tool for quantum chemistry on near-term quantum devices (NISQ era).

📊 Example: H₄ Molecule

In this code, we prepare the ground state of H₄ and estimate the total electronic energy.
✅ Reference state: Hartree–Fock
✅ Ansatz growth: gradient-based operator selection
✅ Energy convergence tracked with ΔE

⚡ Installation
git clone https://github.com/raffrant/ADAPT-VQE.git
cd adapt-vqe-h4
pip install -r requirements.txt

📚 References
  Grimsley, H. R., et al. “An adaptive variational algorithm for exact molecular simulations on a quantum computer.” Nature Communications 10, 3007 (2019).
  Peruzzo, A., et al. “A variational eigenvalue solver on a photonic quantum processor.” Nature Communications 5, 4213 (2014).
