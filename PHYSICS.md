# The Physics, for ML People

A short tour of the ideas behind this project, written for someone who's comfortable with vectors, matrices, and gradient descent but has never had a course in quantum mechanics. No prior physics assumed.

---

## 1. Qubits are unit vectors in $\mathbb{C}^2$

A classical bit is 0 or 1. A **qubit** is a complex unit vector in two dimensions:

$$
|\psi\rangle = \alpha\,|0\rangle + \beta\,|1\rangle, \qquad \alpha, \beta \in \mathbb{C}, \qquad |\alpha|^2 + |\beta|^2 = 1.
$$

Think of $|0\rangle = \begin{pmatrix}1\\0\end{pmatrix}$ and $|1\rangle = \begin{pmatrix}0\\1\end{pmatrix}$ as the two basis vectors. Any unit vector in this plane is a valid qubit state. The squared magnitudes $|\alpha|^2$ and $|\beta|^2$ are the probabilities of seeing 0 or 1 *if you measure* — but until you do, the qubit is genuinely the whole vector, not "secretly" one or the other. This is **superposition**: the linear combination is a real physical state with no classical analogue.

The weird and useful bit: a qubit carries **two real numbers** of continuous information (one complex amplitude up to global phase and normalization), but a measurement only ever returns one classical bit. The information lives in the geometry of the state, not in the measurement outcome of any single experiment.

## 2. $N$ qubits live in $\mathbb{C}^{2^N}$

Two qubits combine via the **tensor product**:

$$
|\psi\rangle \otimes |\phi\rangle \in \mathbb{C}^2 \otimes \mathbb{C}^2 = \mathbb{C}^4.
$$

The basis is $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$. The dimension is $2^N$ — exponential in the number of qubits. This is exactly why simulating quantum systems on classical hardware is hard, and exactly why $N=8$ in this project: $2^8 = 256$ is comfortable for a laptop, $2^{50}$ is not.

A two-qubit state is **entangled** when it can't be written as a tensor product of single-qubit states. The canonical example is

$$
\tfrac{1}{\sqrt{2}}\bigl(|00\rangle + |11\rangle\bigr),
$$

which has no factorization into $|\psi_1\rangle \otimes |\psi_2\rangle$. Entanglement means correlations that no classical joint distribution can reproduce. **For ML intuition:** an entangled state is to a product state what a full joint distribution $p(x,y)$ is to its marginals $p(x)\,p(y)$ — except quantum joint correlations can be even stronger than any classical joint distribution allows.

## 3. Operators are matrices; observables are Hermitian

Anything you can *do* to a state is a matrix acting on it. Anything you can *measure* is a Hermitian matrix $\hat{O} = \hat{O}^\dagger$. The expected value of a measurement is

$$
\langle \hat{O} \rangle = \langle\psi|\,\hat{O}\,|\psi\rangle = \psi^\dagger O \psi
$$

— the same expression you'd write for a quadratic form in linear algebra. Two examples we use everywhere in this project:

- $\hat{\sigma}^z = \begin{pmatrix}1 & 0\\0 & -1\end{pmatrix}$ — measures whether a single qubit is in $|0\rangle$ (eigenvalue $+1$) or $|1\rangle$ (eigenvalue $-1$).
- $\hat{\sigma}^z_i \hat{\sigma}^z_j$ (for two qubits) — measures the *correlation* between qubits $i$ and $j$. Positive means "they tend to agree," negative means "they tend to disagree."

These are the 288 features. We measure $N=8$ single-body $\langle\hat\sigma^z_i\rangle$ and $\binom{8}{2}=28$ two-body $\langle\hat\sigma^z_i\hat\sigma^z_j\rangle$ at each of 8 timesteps.

## 4. The Hamiltonian generates dynamics

The **Hamiltonian** $\hat{H}$ is the Hermitian matrix that encodes the energy of the system. It plays the same role as a vector field in an ODE: it tells you how the state changes with time. The equation of motion is the **Schrödinger equation**:

$$
i\,\frac{d|\psi\rangle}{dt} = \hat{H}\,|\psi\rangle.
$$

For a time-independent $\hat{H}$, this integrates to

$$
|\psi(t)\rangle = e^{-i\hat{H} t}\,|\psi(0)\rangle.
$$

The matrix exponential $U(t) = e^{-i\hat{H} t}$ is **unitary** ($U^\dagger U = I$), so it preserves norm — the state stays on the unit sphere. Unitary evolution is the quantum analogue of a measure-preserving flow on phase space: nothing dissipates.

In this project, every "evolve the system one timestep" is literally a matrix exponential: `U = scipy.linalg.expm(-1j * H * dt)`, then `state = U @ state`.

## 5. The Rydberg Hamiltonian, in plain words

A **Rydberg atom** is an atom whose outermost electron has been bumped into a very high orbit by a laser. Two such atoms interact strongly through their dipole moments — much more strongly than ground-state atoms. This experimental platform (used by companies like QuEra and Pasqal) gives us natural qubits with strong, tunable couplings.

The Hamiltonian we use has three pieces:

$$
\hat{H} = \underbrace{\sum_{i} \tfrac{\Omega}{2}\,\hat{\sigma}^x_i}_{\text{drive}} - \underbrace{\sum_{i} \Delta_i\,\hat{n}_i}_{\text{detuning}} + \underbrace{\sum_{i<j} \frac{C}{|i-j|^6 d^{\,6}}\,\hat{n}_i \hat{n}_j}_{\text{interaction}}
$$

Translating each term:

- **Drive ($\Omega$).** A laser hits every atom equally and rotates each qubit between $|0\rangle$ and $|1\rangle$. Without this, nothing would happen — the drive is what makes the system *go*.
- **Detuning ($\Delta_i$).** A per-atom energy bias. This is the **knob we use to inject classical data**: each atom $i$ gets a $\Delta_i$ equal to the $i$-th PCA component of the input image. Different inputs ⇒ different Hamiltonians ⇒ different evolutions.
- **Interaction ($C$).** Two atoms in the excited Rydberg state repel each other with a $1/r^6$ van der Waals potential. This is what *couples* the atoms and creates entanglement during the dynamics. Set $C=0$ and the atoms evolve independently; turn $C$ on and they don't.

## 6. Why this is good for machine learning

A reservoir computer is a fixed, complicated dynamical system that you drive with input and read out linearly. Three properties make a system a good reservoir:

1. **Rich nonlinearity** — the response to inputs should be highly nonlinear, so distinct inputs land in linearly-separable parts of feature space.
2. **High effective dimensionality** — the reservoir should map a low-dimensional input into a much higher-dimensional internal state.
3. **Fading memory / sensitivity** — past inputs should leave traces in the current state, but not dominate it.

A driven, interacting quantum system delivers all three almost for free:

1. The map from $\Delta$ to $\langle\sigma^z\rangle(t)$ is highly nonlinear (it's the matrix exponential of a Hamiltonian that depends linearly on $\Delta$, then sandwiched between observables — generically very non-linear in $\Delta$).
2. An $N$-qubit system lives in a $2^N$-dimensional Hilbert space. With 8 atoms we expand 8 input numbers into 288 measured features — a 36× expansion.
3. Time evolution of an interacting many-body system mixes information across all atoms within a few steps. By $t=4$, every measured observable depends on every input component, in a way that's effectively impossible to invert by hand.

And critically: **you don't train the reservoir.** No gradients, no parameters, no backprop through the quantum part. The nonlinearity is *given to you for free* by the physics. All the learning happens in a single linear layer on top.

That's the conceptual punchline. A complicated piece of nature is already doing the messy nonlinear work of feature extraction. Our job is just to read it out.

---

*Further reading:* the Wikipedia articles on **qubits**, **quantum entanglement**, **Hamiltonian (quantum mechanics)**, and **Rydberg atom** are all surprisingly readable and sufficient to flesh out anything compressed above.
