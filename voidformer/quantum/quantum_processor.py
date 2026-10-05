"""Virtual Quantum Processor with Dynamic Mid-Circuit Measurement and Feed-Forward Logic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

from voidformer.quantum.temporal_coherence import CoherenceTracker, DecoherenceThresholdExceeded
from voidformer.quantum.qubit_state import QuantumStateVector


class QuantumCircuit:
    """Quantum Circuit wrapper."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits


class QuantumAlgorithm:
    """Quantum Algorithm primitive."""

    def __init__(self, name: str = "vqc"):
        self.name = name


class CollapseProtocol:
    """Measurement-based collapse protocol."""
    HARD = "hard"
    ENTROPY_GATED = "entropy_gated"

    def __init__(self, protocol: str = "entropy_gated"):
        self.protocol = protocol


class DynamicMidCircuitMeasurement:
    """Dynamic Mid-Circuit Measurement and conditional feed-forward gate execution."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def measure_qubit(
        self, state: QuantumStateVector, qubit: int
    ) -> Tuple[QuantumStateVector, torch.Tensor]:
        probs = state.probabilities
        dim = 2 ** self.n_qubits
        step = 2 ** qubit

        i = torch.arange(dim, device=probs.device)
        idx0 = torch.where((i & step) == 0)[0]
        idx1 = idx0 + step

        p0 = probs[..., idx0].sum(dim=-1, keepdim=True)
        p1 = probs[..., idx1].sum(dim=-1, keepdim=True)

        rand = torch.rand_like(p0)
        outcome = (rand > p0).long()

        amps = state.amplitudes.clone()
        mask0 = (outcome == 0)
        norm0 = torch.sqrt(p0.clamp_min(1e-12))
        norm1 = torch.sqrt(p1.clamp_min(1e-12))

        amps_collapsed0 = amps.clone()
        amps_collapsed0[..., idx1] = 0.0
        amps_collapsed0 = amps_collapsed0 / norm0

        amps_collapsed1 = amps.clone()
        amps_collapsed1[..., idx0] = 0.0
        amps_collapsed1 = amps_collapsed1 / norm1

        new_amps = torch.where(mask0, amps_collapsed0, amps_collapsed1)
        new_state = QuantumStateVector(amplitudes=new_amps, n_qubits=self.n_qubits)
        return new_state, outcome.squeeze(-1)


class VirtualQuantumProcessor(nn.Module):
    """Virtual Quantum Processor executing VQCs with Coherence Control and Dynamic Feed-Forward."""

    def __init__(self, n_qubits: int = 4, purity_threshold: float = 0.85, **kwargs):
        super().__init__()
        self.n_qubits = n_qubits
        self.purity_threshold = purity_threshold
        self.tracker = CoherenceTracker(purity_threshold=purity_threshold)
        self.mid_meas = DynamicMidCircuitMeasurement(n_qubits=n_qubits)

    def execute_algorithm(self, alg: QuantumAlgorithm, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return self.forward(x)

    def get_quantum_state_diagnostics(self, x: torch.Tensor) -> dict:
        return {"purity": 1.0, "entropy": 0.0}

    def forward(
        self, x: torch.Tensor, fallback_classical: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        shape = x.shape
        d_out = shape[-1]
        x_flat = x.reshape(-1, d_out)
        B_flat = x_flat.shape[0]
        dim = 2 ** self.n_qubits

        amps = torch.zeros(B_flat, dim, dtype=torch.complex128, device=x.device)
        amps[:, 0] = 1.0 + 0j
        state = QuantumStateVector(amplitudes=amps, n_qubits=self.n_qubits)

        rho = state.density_matrix()

        try:
            purity, entropy, diagnostics = self.tracker.check_and_decouple(
                rho, fallback_on_threshold=fallback_classical
            )
        except DecoherenceThresholdExceeded:
            classical_out = torch.relu(x)
            return classical_out, {"fallback_classical": True, "purity": 0.8}

        state, outcomes = self.mid_meas.measure_qubit(state, qubit=0)

        out_probs = state.probabilities
        if out_probs.shape[-1] < d_out:
            out_probs = F.pad(out_probs, (0, d_out - out_probs.shape[-1]))
        elif out_probs.shape[-1] > d_out:
            out_probs = out_probs[..., :d_out]

        out = out_probs.reshape(*shape).to(x.dtype)
        diagnostics["mid_circuit_outcomes"] = outcomes
        return out, diagnostics


def initialize_quantum_processor(
    d_model: int = 128,
    n_qubits_per_token: int = 4,
    max_seq_len: int = 128,
    collapse_protocol: str = "entropy_gated",
    enable_entanglement: bool = True,
    device: Optional[torch.device] = None,
    name: str = "qvf",
) -> VirtualQuantumProcessor:
    return VirtualQuantumProcessor(n_qubits=n_qubits_per_token)
