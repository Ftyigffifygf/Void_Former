# VoidFormer: Virtual Quantum Computing Simulator & Hybrid Quantum-Classical QML

VoidFormer is a research-style, PyTorch-based **quantum computing simulator** with a
transformer-style model whose "quantum" layers are **simulated on classical hardware
(CPU/GPU)**. It also includes an **IBM Quantum bridge** (Qiskit Runtime) so selected
circuits can be run on real quantum hardware, and a small **hybrid quantum-classical ML**
module (PennyLane + PyTorch).

> **What this project is, and is not**
> - It **is** a simulator, a circuit toolkit, and a hybrid QML experiment platform.
> - It is **not** an LLM running on a quantum computer. Transformer training and
>   inference run in PyTorch on classical hardware. Only small circuits (a few qubits)
>   are sent to IBM hardware.

## Core ideas

1. **Superposition**: tokens are state vectors `|ψ⟩ = Σᵢ αᵢ|i⟩` in a 2^n Hilbert space.
2. **Entanglement**: correlations between tokens via CNOT, Bell and GHZ states.
3. **Quantum gates**: unitary operators (H, X, Y, Z, CNOT, Toffoli, Phase, T).
4. **Subordinate Hybrid Layer**: routes computation between quantum state space and classical neural layers.
5. **Data Re-uploading & QFT Maps**: interleaved input re-uploading and QFT phase feature maps.
6. **Measurement**: Born-rule collapse `P(i) = |αᵢ|²` converts quantum state to classical output.

```
Classical:  |T⟩ = α|S_c⟩ + β|S_v⟩ + γ·I(|S_c⟩,|S_v⟩)
            ↓
Quantum:    |ψ⟩ = Σᵢ αᵢ|i⟩   where αᵢ ∈ ℂ, Σ|αᵢ|² = 1
            ↓ [Quantum gates: U|ψ⟩]
            ↓ [Entanglement: CNOT]
            ↓ [Data Re-uploading & QFT Map]
            ↓ [Subordinate Hybrid Layer]
            ↓ [Measurement: collapse]
Classical:  observed state with P(i) = |αᵢ|²
```

---

## Unified Multi-Backend Quantum Execution

VoidFormer provides execution bridges and optional wrappers across multiple quantum software development kits:

| Backend | Framework / SDK | Description | Status |
|---|---|---|---|
| **PyTorch Virtual** | PyTorch Native | High-performance CPU/GPU quantum statevector simulation | ✅ Active / Native |
| **IBM Quantum / Qiskit** | `qiskit_ibm_runtime` | IBM QPU & Aer noisy simulator execution bridge | ✅ Integrated |
| **PennyLane** | `pennylane` | Automatic differentiation QML circuits and PyTorch QNN layers | ✅ Integrated |
| **Paddle Quantum** | `paddle_quantum` | Optional PaddlePaddle PQC variational circuit bridge | ⚠️ Optional, untested |
| **CUDA-Q** | `cudaq` | Optional NVIDIA GPU-accelerated quantum simulation bridge | ⚠️ Optional, untested |
| **Microsoft Quantum** | `qsharp` / QDK | Optional Microsoft QDK Azure Quantum Resource Estimator bridge | ⚠️ Optional, untested |

*Note: `qml/` modules and benchmark scripts should be executed from the repository root.*

---

## IBM Quantum hardware results (October 2026)

Circuits built with VoidFormer's gate lists were converted with Qiskit, transpiled to
IBM's native gates (`rz`, `sx`, `cz`), and run on real IBM Quantum hardware through
Qiskit Runtime. Each hardware circuit ran **4096 shots**; the local simulator reference
used 1024 shots.

| Circuit | Transpiled depth | Hardware result | Hellinger fidelity vs simulator |
|---|---|---|---|
| GHZ, 3 qubits | 12 | success rate **0.928** | 0.928 |
| GHZ, 4 qubits | 16 | success rate **0.856** | 0.852 |
| GHZ, 5 qubits | 20 | success rate **0.819** | 0.815 |
| Grover, 2 qubits | 12 | target state `11` found in **91.6%** of shots | 0.916 |
| QFT, 3 qubits | 41 | see note below | 0.994 |
| QFT, 4 qubits | 101 | see note below | 0.991 |

*GHZ success rate = probability of measuring all-zeros or all-ones.*

**Reading these results**

- GHZ success falls as qubits are added (0.928 → 0.856 → 0.819), the expected trend
  because each extra CNOT adds noise. In the 4- and 5-qubit runs, `00…0` appears more
  often than `11…1`, and the common errors are 1→0 bit flips, which is consistent with
  qubit relaxation and readout error.
- **QFT note:** the ideal output of a QFT on `|0…0⟩` is a uniform distribution, and heavy
  hardware noise also produces a near-uniform distribution. The high fidelities therefore
  do **not** show the QFT ran correctly. A meaningful test needs a non-trivial input state
  or a QFT followed by an inverse QFT (planned).
- Hardware job IDs: GHZ/Grover/QFT batch `db0v73avog1s73fhd1hg`; VQC evaluation on
  `ibm_fez`: `db0vapqvog1s73fhd5fg`. The full runs and printed outputs are in
  [`void-former-quantum.ipynb`](void-former-quantum.ipynb).

---

## Hybrid quantum machine learning (`qml/`)

`qml/` contains a PennyLane + PyTorch hybrid module, always benchmarked against a
parameter-matched classical baseline over multiple seeds:

- **HybridNet vs ClassicalNet** (`qml/layers.py`, `qml/train.py`): a small variational
  quantum circuit (angle embedding + trainable entangling layers) inside a PyTorch
  network, on a binary text-sentiment task (TF-IDF + SVD features).
- **Hybrid quantum language model** (`qml/lm.py`, `qml/train_lm.py`): a small
  character-level transformer whose feed-forward block can be replaced by a variational
  quantum circuit, compared against a classical transformer by perplexity.
- **Backends** (`qml/backends.py`): `default.qubit`, local Qiskit Aer, or IBM devices.
- **Hardware / noisy-simulator evaluation** (`qml/evaluate_ibm.py`): defaults to a local
  Aer simulator; real hardware runs only when `--backend ibm_hardware` is passed, with
  sample and shot caps to protect the free-tier quota.

**Current status: experimental.** In the recorded simulator run, the hybrid classifier's
loss stayed near chance (about 0.684 against 0.693 for a binary task) over 15 epochs
while the classical baseline improved (0.659 → 0.616). The hybrid model has not yet
shown that it learns, and no quantum advantage is claimed.

---

## Known limitations (read before citing any number)

- **Simulator, not a quantum LLM.** The language model and its "quantum" layers run in
  PyTorch. Real hardware is used for small circuits only.
- **The hardware-in-the-loop demo in the notebook does not train the quantum circuit.**
  In `HardwareQuantumLayer.forward`, the circuit angles are detached from the autograd
  graph, so they receive no gradients and stay fixed (identical from epoch 1). The loss
  decrease in that demo comes from the classical linear layers fitting 20 randomly
  generated samples with random labels. The demo shows that a PyTorch model can call a
  real QPU on every forward pass, not that quantum parameters were trained on hardware.
- **Hellinger fidelity on near-uniform outputs is not a quality metric.** The VQC
  evaluation (fidelity 0.9974 on `ibm_fez`) compared two nearly uniform distributions.
  Report test accuracy on held-out data instead.
- **Free-tier limits.** IBM's Open Plan allows about 10 minutes of quantum runtime per
  28 days (with an optional one-time larger promotion). Training with hardware in the
  loop is limited to tiny circuits.

---

## Quick start

```bash
# 1. Test the quantum processor
python -m voidformer.quantum_init

# 2. Run the multi-backend quantum test suite
pytest tests/test_subordinate_hybrid.py

# 3. Hybrid QML benchmark (HybridNet vs classical baseline, multiple seeds)
python -m qml.train --epochs 15 --seeds 42 43 44

# 4. DeepSeek Quantum Multi-Backend Evaluation Harness
python -c "from voidformer.harness.cli import run_deepseek_quantum_eval; print(run_deepseek_quantum_eval())"
```

---

## Features at a glance

| Component | Description | Status |
|-----------|-------------|--------|
| Multi-Backend Registry | Unified bridge for Qiskit, PennyLane, Paddle Quantum, CUDA-Q, MS QDK | ✅ |
| Subordinate Hybrid Layer | Quantum / Classical Workload Split engine | ✅ |
| Data Re-uploading QNN | Data re-uploading quantum circuit layer | ✅ |
| QFT Feature Map | Quantum Fourier Transform phase mapping | ✅ |
| QSRE Engine | Quantum Superposition Reasoning Engine (Hilbert latent thinking) | ✅ |
| Quantum MoE | Superposition Mixture of Experts with fidelity routing | ✅ |
| DeepSeek Harness | DeepSeek R1/V3 quantum evaluation & QDK resource estimation | ✅ |
| IBM Backend Bridge | Qiskit Runtime bridge (`voidformer/quantum/ibm_backend.py`) | ✅ |

---

## Citation

```
@misc{voidformer_quantum2026,
  title  = {VoidFormer: A Virtual Quantum Computing Simulator for Neural Language Models},
  year   = {2026},
  note   = {Quantum superposition, entanglement, and multi-backend subordinate hybrid neural architecture}
}
```

## License

See [LICENSE](LICENSE).
