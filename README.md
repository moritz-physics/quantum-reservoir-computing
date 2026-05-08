# Quantum Reservoir Computing on a Simulated Rydberg Atom Array

**Classifying MNIST digits with the natural dynamics of an 8-atom quantum system.**

This project implements a Quantum Reservoir Computer (QRC) from scratch in NumPy/PyTorch. Classical handwritten-digit images are encoded as detuning fields on a one-dimensional chain of simulated Rydberg atoms; the quantum state is evolved under the resulting many-body Hamiltonian; and the trajectory of measured observables becomes a high-dimensional feature vector that a tiny linear layer can read out. No variational quantum circuit, no parameterized gates — the reservoir itself is fixed, and only the classical readout is trained.

> *Originally developed as part of the Quantum Computing and Machine Learning lab course at Ludwig Maximilian University of Munich (LMU). Migrated here for portfolio purposes.*

---

## Why this is interesting

Most "quantum machine learning" approaches train a parameterized circuit. Reservoir computing flips that idea on its head: take a complex, untrained dynamical system, drive it with your input, and let its rich nonlinear response do the feature engineering for you. A linear layer on top is enough.

Rydberg atom arrays are an unusually good substrate for this:

- **Strong, tunable interactions.** Pairs of atoms in highly excited Rydberg states interact via a $1/r^6$ van der Waals potential, so neighboring atoms are strongly coupled and distant atoms aren't.
- **Programmable detuning.** Per-atom laser detunings $\Delta_i$ are knobs we can directly use to inject classical data.
- **Genuine many-body dynamics.** With $N=8$ atoms the Hilbert space has $2^8 = 256$ dimensions — small enough to simulate exactly on a laptop, large enough that the dynamics are non-trivial and entangling.

If interactions matter for the task, turning them off should hurt. That's exactly the comparison this notebook runs.

---

## The Hamiltonian

The system evolves under the standard Rydberg Hamiltonian:

$$
\hat{H} = \underbrace{\sum_{i} \frac{\Omega}{2}\,\hat{\sigma}^x_i}_{\text{drive}} \;-\; \underbrace{\sum_{i} \Delta_i\,\hat{n}_i}_{\text{detuning}} \;+\; \underbrace{\sum_{i<j} \frac{C}{|i-j|^6\, d^{\,6}}\,\hat{n}_i \hat{n}_j}_{\text{van der Waals interaction}}
$$

where $\hat{n}_i = (I - \hat{\sigma}^z_i)/2$ projects onto the Rydberg-excited state of atom $i$, $\Omega$ is the Rabi frequency of the global drive, $\Delta_i$ is the per-atom detuning (this is where the data is injected), $d$ is the atomic spacing, and $C$ is the interaction coefficient.

- $C = 0$ → atoms are independent. The reservoir reduces to $N$ decoupled qubits.
- $C \neq 0$ → atoms entangle through the interaction term, and the dynamics become genuinely many-body.

---

## Pipeline

```
MNIST image  ──►  flatten + normalize  ──►  PCA (8 components)  ──►  rescale to [-6, 6]
                                                                        │
                                                                        ▼
                                                    detunings  Δ₁, …, Δ₈  (one per atom)
                                                                        │
                                                                        ▼
                                       build Ĥ(Δ),  evolve  |ψ(t+Δt)⟩ = e^{-iĤΔt} |ψ(t)⟩
                                                                        │
                                              measure ⟨σᶻᵢ⟩ and ⟨σᶻᵢσᶻⱼ⟩ at every time step
                                                                        │
                                                                        ▼
                                          288-dimensional feature vector  (8 timesteps × 36 obs.)
                                                                        │
                                                                        ▼
                                            linear classifier  →  predicted digit (0–9)
```

**Step by step:**

1. **Data encoding.** Each $28\times28$ MNIST image is flattened, PCA-reduced to 8 components, and rescaled to $[-6, 6]\,\text{rad}/\mu\text{s}$. The 8 components become the 8 atomic detunings — one knob per atom.
2. **Hamiltonian construction.** All operators are built explicitly via Kronecker products, no quantum SDK. $\Omega = 2\pi\,\text{MHz}$, $d = 10\,\mu\text{m}$.
3. **Time evolution.** The system starts in $|0\rangle^{\otimes 8}$ and is evolved for 8 steps of $\Delta t = 0.5$ via the matrix exponential $U = e^{-i\hat{H}\Delta t}$ (using `scipy.linalg.expm`).
4. **Observables.** At each timestep we measure all $N = 8$ single-body $\langle\hat{\sigma}^z_i\rangle$ and all $\binom{N}{2} = 28$ two-body correlations $\langle\hat{\sigma}^z_i\hat{\sigma}^z_j\rangle$. That's 36 numbers per step × 8 steps = **288-dimensional features per image**.
5. **Readout.** A single `nn.Linear(288, 10)` is trained with cross-entropy and Adam. (Plus baselines and a 4-layer MLP for comparison.)

The training subset is intentionally small (1000 train / 200 test) — the point isn't to win MNIST, it's to compare what the quantum dynamics buy you on a fixed budget of classical data.

---

## What the experiments compare

| # | Model | Features | What it tests |
|---|---|---|---|
| 1 | Linear | QRC, $C = 0$ | Quantum reservoir without interactions — 8 independent driven qubits |
| 2 | Linear | QRC, $C = 10^7$ | Full Rydberg reservoir with van der Waals coupling |
| 3 | Linear | Raw 8-dim PCA | "Did we need the quantum part at all?" baseline |
| 4 | 4-layer MLP | Raw 8-dim PCA | "Could a deeper *classical* head on the same input do as well?" |
| ★ | Optuna-tuned MLP | Raw 8-dim PCA | Hyperparameter-optimized classical baseline |

### Findings

- **Interactions help.** The interacting reservoir (Exercise 2, $C \neq 0$) outperforms the non-interacting one (Exercise 1, $C = 0$) under the same linear readout. With $C = 0$ the 8 atoms evolve independently, so the two-body correlators $\langle\sigma^z_i\sigma^z_j\rangle$ factorize into products of one-body terms — the 288 features carry no genuine joint information beyond what 8 independent oscillators already contain. Turning interactions on lifts that constraint and lets the reservoir produce truly coupled features.
- **A linear head on quantum features beats a linear head on raw PCA.** This is the central QRC claim demonstrated on a small problem: the quantum dynamics expand 8 numbers into a richer representation that a linear classifier can separate.
- **A deep MLP on raw PCA closes most of the gap.** Unsurprising — a 4-layer network is a universal approximator on this tiny 8-dim input, so it can learn its own nonlinear features. The QRC's appeal isn't that it beats a deep net; it's that a *fixed*, untrained quantum system plus a linear layer reaches comparable territory.
- **Optuna's tuned MLP** sets the practical ceiling on what the 8 PCA components alone can deliver.

The exact accuracy numbers are reported in the plot titles inside the notebook (re-running reproduces them); the qualitative ordering is robust.

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# clone and enter
git clone <this-repo>
cd quantum-reservoir-computing

# install dependencies (Python 3.12)
uv sync

# launch JupyterLab
uv run jupyter lab Mori_qrc_classification.ipynb
```

Dependencies: `numpy`, `scipy`, `torch`, `scikit-learn`, `matplotlib`, `tqdm`, `optuna`, `jupyterlab`. MNIST downloads automatically on first run (~12 MB).

Runtime: the two quantum embedding passes (1000 train + 200 test images, twice) take ~2.5 minutes total on a laptop CPU. Everything else is fast.

---

## Files

- `Mori_qrc_classification.ipynb` — the entire project: Hamiltonian construction, simulation, embedding, training, comparisons, hyperparameter tuning.
- `PHYSICS.md` — a short, ML-friendly tour of the quantum-mechanics ideas behind the project. Read this first if "Hamiltonian" and "entanglement" feel hand-wavy.
- `README.md` — you are here.

---

## References

- **Fujii & Nakajima (2017).** *Harnessing disordered quantum dynamics for machine learning.* [arXiv:2011.04890](https://arxiv.org/abs/2011.04890) — the foundational QRC paper showing that natural quantum dynamics make excellent reservoirs.
- **Burgess & Florescu (2022).** *Quantum reservoir computing implementations for classical and quantum problems.* [arXiv:2211.08567](https://arxiv.org/abs/2211.08567) — analyses of QRC architectures including Rydberg-style platforms; closest match to the setup used here.

---

*Built as a lab exercise at LMU Munich. Kept here because it's a clean, self-contained illustration of an idea I find genuinely beautiful: that a complicated physical system, left to its own devices, is already doing useful computation — you just have to read it out.*
