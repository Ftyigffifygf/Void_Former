# Quantum VoidFormer Complexity & Scaling Analysis

This document provides formal time, space, and memory complexity bounds for the **VoidFormer Virtual Quantum Computing Simulator**, detailing scaling behavior and computational limits across qubit counts ($n$) and sequence lengths ($T$).

---

## 1. Hilbert Space State Vector Scaling

Each token in a sequence of length $T$ is represented as a complex state vector in Hilbert space $\mathcal{H} = \mathbb{C}^{2^n}$, where $n$ is the number of qubits per token.

| Component | State Representation | Space Complexity |
| :--- | :--- | :--- |
| **Per-Token Quantum State** | $|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle, \quad \alpha_i \in \mathbb{C}$ | $\mathcal{O}(2^n)$ |
| **Sequence Quantum State** | Batch of $B$ sequences of length $T$ | $\mathcal{O}(B \cdot T \cdot 2^n)$ |
| **Shared Cross-Token Register** | Global register $Z_{\text{shared}} \in \mathbb{C}^{B \times 2^{n_{\text{shared}}}}$ | $\mathcal{O}(B \cdot 2^{n_{\text{shared}}})$ |

---

## 2. Gate Application Time Complexity

Applying an $m$-qubit gate operator $U \in \mathbb{C}^{2^m \times 2^m}$ to $m$ specified target qubits within an $n$-qubit token register utilizes tensor reshaping and dimension permuting:

1. **Reshape & Permute**: Splits $(B, T, 2^n)$ into $(B, T, 2, \dots, 2)$ ($n$ axes of size 2) and permutes target qubits to trailing axes: $\mathcal{O}(B \cdot T \cdot 2^n)$ time.
2. **Matrix Contraction**: Batched matrix multiplication flat state $(B, T, 2^{n-m}, 2^m) \times U^T (2^m, 2^m)$: $\mathcal{O}(B \cdot T \cdot 2^{n-m} \cdot 2^{2m}) = \mathcal{O}(B \cdot T \cdot 2^{n+m})$ FLOPs.
3. **Inverse Permute**: Restores original axis layout: $\mathcal{O}(B \cdot T \cdot 2^n)$.

$$\text{Total Gate Time Complexity: } \mathcal{O}\left(B \cdot T \cdot 2^{n+m}\right)$$

For $1$-qubit gates ($m=1$), complexity is $\mathcal{O}(B \cdot T \cdot 2^{n+1})$.
For $2$-qubit gates ($m=2$), complexity is $\mathcal{O}(B \cdot T \cdot 2^{n+2})$.

---

## 3. Entanglement & Measurement Complexity

| Operation | Mathematical Mechanism | Time Complexity |
| :--- | :--- | :--- |
| **Pairwise Entanglement** | Pairwise CNOT gates over $K$ token pairs | $\mathcal{O}(K \cdot B \cdot 2^{n+2})$ |
| **Cross-Token Entanglement** | Token-to-Shared projection and coupling | $\mathcal{O}\left(B \cdot T \cdot 2^n \cdot 2^{n_{\text{shared}}}\right)$ |
| **Hard Collapse (STE / Gumbel)** | Born rule probability calculation $P(i) = \|\alpha_i\|^2$ + STE sampling | $\mathcal{O}(B \cdot T \cdot 2^n)$ |
| **Quantum Kernel Attention** | Quantum state overlap $K(x,y) = \|\langle \psi(x) \| \psi(y) \rangle\|^2$ | $\mathcal{O}(B \cdot T^2 \cdot 2^n)$ |

---

## 4. Practical Limits and Scaling Thresholds

Standard PyTorch simulation on standard hardware operates under the following resource constraints:

- **$n \le 10$ Qubits per Token**: $2^{10} = 1024$ dimensions per token. Highly fast and fits easily in standard GPU memory.
- **$n \approx 14\text{--}16$ Qubits per Token**: $2^{16} = 65,536$ dimensions per token. Achievable on GPUs with $\ge 24\text{ GB}$ VRAM.
- **$n > 20$ Qubits per Token**: Full state vector simulation hits memory limits ($\ge 1\text{M}$ complex amplitudes per token). Mitigation via Matrix Product States (MPS) / Tensor Train compression in `quantum/qiml.py` compresses state representations to bond dimension $\chi$:

$$\text{MPS Compressed Space Complexity: } \mathcal{O}(n \cdot d \cdot \chi^2)$$
