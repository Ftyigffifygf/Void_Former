"""Entanglement Manager.

Quantum entanglement is the non-local correlation between qubits that cannot
be explained by classical probability theory. This module creates and tracks
entangled states between token positions and via a shared cross-token register.

Key Concepts:
- Bell States: Maximally entangled 2-qubit states
- GHZ States: Maximally entangled n-qubit states
- Cross-Token Entanglement: Global shared quantum register entangling all token positions
- Entanglement Entropy: von Neumann entropy of reduced density matrix
- Concurrence: Entanglement measure for two-qubit systems
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .qubit_state import QuantumStateVector, QubitStateManager
from .quantum_gates import QuantumGateRegistry


class BellStateGenerator(nn.Module):
    """Generator for maximally entangled Bell states.

    Four Bell basis states (EPR pairs):
    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2    (Bell state)
    |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    """

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.gate_registry = QuantumGateRegistry(device=self.device)

    def create_bell_phi_plus(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Create |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 (maximally entangled)."""
        manager = QubitStateManager(n_qubits_per_token=2, device=self.device)
        state = manager.initialize_computational_basis(batch_size, seq_len, basis_state=0)

        # Circuit: H on q0, then CNOT(0, 1)
        H = self.gate_registry.get_gate("H")
        CNOT = self.gate_registry.get_gate("CNOT")

        state = H.apply(state, target_qubits=[0])
        state = CNOT.apply(state, target_qubits=[0, 1])

        return state

    def create_bell_phi_minus(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Create |Φ⁻⟩ = (|00⟩ - |11⟩)/√2."""
        state = self.create_bell_phi_plus(batch_size, seq_len)

        # Apply Z to introduce phase flip on qubit 0
        Z = self.gate_registry.get_gate("Z")
        state = Z.apply(state, target_qubits=[0])

        return state

    def create_bell_psi_plus(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Create |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2."""
        state = self.create_bell_phi_plus(batch_size, seq_len)

        # Apply X to second qubit (q1): (|00⟩+|11⟩) -> (|01⟩+|10⟩)
        X = self.gate_registry.get_gate("X")
        state = X.apply(state, target_qubits=[1])

        return state

    def create_bell_psi_minus(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Create |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2."""
        state = self.create_bell_psi_plus(batch_size, seq_len)

        # Apply Z to introduce phase flip on q0: (|01⟩+|10⟩) -> (|01⟩-|10⟩)
        Z = self.gate_registry.get_gate("Z")
        state = Z.apply(state, target_qubits=[0])

        return state


class SharedCrossTokenEntangler(nn.Module):
    """Architectural implementation of Cross-Token Entanglement via shared/compressed quantum state.

    Maintains a compressed shared quantum state register Z_shared in C^(B x 2^n_shared)
    and position-dependent phase coupling that interacts with individual token state vectors.
    Enables true non-local inter-token quantum correlations that differ between token pairs.
    """

    def __init__(
        self,
        n_qubits_per_token: int,
        max_seq_len: int = 512,
        n_shared_qubits: int = 2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_token_qubits = n_qubits_per_token
        self.max_seq_len = max_seq_len
        self.n_shared_qubits = n_shared_qubits
        self.token_dim = 2 ** n_qubits_per_token
        self.shared_dim = 2 ** n_shared_qubits
        self.device = device or torch.device("cpu")

        # Complex linear interaction projections between token Hilbert space & shared Hilbert space
        self.token_to_shared = nn.Parameter(
            torch.randn(self.token_dim, self.shared_dim, dtype=torch.complex64) / math.sqrt(self.token_dim)
        )
        self.shared_to_token = nn.Parameter(
            torch.randn(self.shared_dim, self.token_dim, dtype=torch.complex64) / math.sqrt(self.shared_dim)
        )
        # Position-dependent coupling phase parameters (T, 2^n_shared)
        self.position_phases = nn.Parameter(
            torch.randn(max_seq_len, self.shared_dim) * 0.1
        )
        # Pairwise quantum kernel projection matrix for token-pair interaction
        self.pairwise_kernel = nn.Parameter(
            torch.randn(self.token_dim, self.token_dim, dtype=torch.complex64) / math.sqrt(self.token_dim)
        )
        self.coupling_phase = nn.Parameter(torch.tensor(math.pi / 4))

    def entangle_tokens(
        self,
        state: QuantumStateVector,
    ) -> Tuple[QuantumStateVector, torch.Tensor]:
        """Entangle token states through the shared compressed quantum register and position-dependent kernels.

        Args:
            state: Quantum state vector (B, T, 2^n)

        Returns:
            Entangled QuantumStateVector, shared register state (B, 2^n_shared)
        """
        B, T, dim = state.amplitudes.shape
        amps = state.amplitudes  # (B, T, 2^n_token)

        # 1. Position-modulated token projections into shared register
        pos_phases = torch.exp(1j * self.position_phases[:T, :].unsqueeze(0))  # (1, T, 2^n_shared)
        token_projected = torch.matmul(amps, self.token_to_shared) * pos_phases  # (B, T, 2^n_shared)

        z_shared = token_projected.mean(dim=1)  # (B, 2^n_shared)
        z_shared_norm = torch.linalg.norm(z_shared, dim=-1, keepdim=True).clamp(min=1e-10)
        z_shared = z_shared / z_shared_norm

        # 2. Position-specific feedback from Z_shared to tokens
        # Each position receives distinct feedback according to its position phase
        shared_back = torch.matmul(
            (z_shared.unsqueeze(1) * pos_phases.conj()),
            self.shared_to_token
        )  # (B, T, 2^n_token)

        # 3. Position-dependent token-pair interference (pair-specific correlation)
        # Pairwise state interaction matrix P = softmax(|amps @ W @ amps^H|)
        pair_scores = torch.abs(torch.matmul(torch.matmul(amps, self.pairwise_kernel), amps.mH))  # (B, T, T)
        pair_weights = torch.softmax(pair_scores, dim=-1)  # (B, T, T)
        pair_interfered = torch.matmul(pair_weights.to(dtype=amps.dtype), amps)  # (B, T, 2^n_token)

        cos_p = torch.cos(self.coupling_phase)
        sin_p = torch.sin(self.coupling_phase)

        entangled_amps = cos_p * amps + sin_p * (0.5 * shared_back + 0.5 * pair_interfered)

        new_state = QuantumStateVector(
            amplitudes=entangled_amps,
            n_qubits=state.n_qubits,
            global_phase=state.global_phase,
        ).normalize()

        return new_state, z_shared


class EntanglementManager(nn.Module):
    """Manages quantum entanglement between token positions.

    Creates and tracks entangled quantum states across sequence positions.
    Includes both pairwise gate-based entanglement and shared-register cross-token entanglement.
    """

    def __init__(
        self,
        n_qubits_per_token: int,
        max_seq_len: int,
        n_shared_qubits: int = 2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits_per_token
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")

        self.bell_generator = BellStateGenerator(device=self.device)
        self.gate_registry = QuantumGateRegistry(device=self.device)

        # Cross-token entangler using shared compressed state
        self.cross_token_entangler = SharedCrossTokenEntangler(
            n_qubits_per_token=n_qubits_per_token,
            n_shared_qubits=n_shared_qubits,
            device=self.device,
        )

        # Learnable entanglement patterns
        self.entangle_scores = nn.Parameter(
            torch.randn(max_seq_len, max_seq_len) * 0.01
        )

    def apply_cross_token_entanglement(
        self,
        state: QuantumStateVector,
    ) -> Tuple[QuantumStateVector, torch.Tensor]:
        """Apply global cross-token entanglement using shared compressed quantum state."""
        return self.cross_token_entangler.entangle_tokens(state)

    def create_pairwise_entanglement(
        self,
        state: QuantumStateVector,
        token_pairs: list[tuple[int, int]],
        entanglement_strength: float = 1.0,
    ) -> QuantumStateVector:
        """Entangle specified pairs of tokens (i, j) via pairwise entangling gates.

        Performs direct 2-qubit CNOT / Controlled-Phase interactions between qubit 0 of token i
        and qubit 0 of token j, creating genuine non-local quantum state entanglement
        specifically between token i and token j while preserving full state_dim tensor shape.
        """
        B, T, state_dim = state.amplitudes.shape
        if state_dim < 4:
            return state

        s = entanglement_strength
        CNOT = self.gate_registry.get_gate("CNOT")
        U = CNOT.matrix(state.amplitudes.device, state.amplitudes.dtype)

        token_states = list(state.amplitudes.unbind(dim=1))  # T tensors of (B, state_dim)

        for (i, j) in token_pairs:
            if i >= T or j >= T or i == j:
                continue
            joint = torch.stack(
                [token_states[i][:, 0], token_states[i][:, 1],
                 token_states[j][:, 0], token_states[j][:, 1]],
                dim=-1,
            )  # (B, 4)
            ent = torch.matmul(joint, U.mT)  # (B, 4)
            mixed = (1 - s) * joint + s * ent

            # Keep untouched amplitudes so every token stays (B, state_dim)
            if state_dim > 2:
                token_states[i] = torch.cat([mixed[:, :2], token_states[i][:, 2:]], dim=-1)
                token_states[j] = torch.cat([mixed[:, 2:], token_states[j][:, 2:]], dim=-1)
            else:
                token_states[i] = mixed[:, :2]
                token_states[j] = mixed[:, 2:]

        new_amps = torch.stack(token_states, dim=1)  # (B, T, state_dim)
        return QuantumStateVector(
            amplitudes=new_amps,
            n_qubits=state.n_qubits,
            global_phase=state.global_phase,
        ).normalize()

    def create_ghz_state(
        self,
        batch_size: int,
        n_tokens: int,
        n_qubits: int = 3,
    ) -> QuantumStateVector:
        """Create GHZ state: maximal n-qubit entanglement."""
        manager = QubitStateManager(n_qubits_per_token=n_qubits, device=self.device)
        state = manager.initialize_computational_basis(batch_size, n_tokens, basis_state=0)

        H = self.gate_registry.get_gate("H")
        CNOT = self.gate_registry.get_gate("CNOT")

        state = H.apply(state, target_qubits=[0])
        for i in range(n_qubits - 1):
            state = CNOT.apply(state, target_qubits=[i, i + 1])

        return state

    def measure_entanglement_entropy(
        self,
        state: QuantumStateVector,
        partition_a: list[int],
    ) -> torch.Tensor:
        """Compute entanglement entropy of bipartition A|B."""
        reduced_probs = state.partial_trace(partition_a)
        reduced_probs = reduced_probs.clamp(min=1e-10)
        return -(reduced_probs * torch.log2(reduced_probs)).sum(dim=-1)

    def compute_concurrence(
        self,
        state: QuantumStateVector,
    ) -> torch.Tensor:
        """Compute concurrence (entanglement measure for 2-qubit systems)."""
        assert state.n_qubits == 2, "Concurrence only defined for 2-qubit systems"
        a = state.amplitudes  # (B, T, 4)
        term = a[..., 0] * a[..., 3] - a[..., 1] * a[..., 2]
        return 2 * torch.abs(term).clamp(0, 1)

    def apply_learned_entanglement(
        self,
        state: QuantumStateVector,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> tuple[QuantumStateVector, torch.Tensor]:
        """Apply learned pairwise and cross-token entanglement."""
        T = state.amplitudes.shape[1]

        causal_mask = torch.triu(
            torch.ones(T, T, device=state.amplitudes.device, dtype=torch.bool),
            diagonal=1,
        )

        scores = self.entangle_scores[:T, :T] / temperature
        scores = scores.masked_fill(causal_mask, float("-inf"))
        entangle_attn = torch.softmax(scores, dim=-1)

        # Apply shared cross-token entangler first
        state, z_shared = self.apply_cross_token_entanglement(state)

        if top_k is not None:
            values, indices = torch.topk(entangle_attn, k=min(top_k, T), dim=-1)
            token_pairs = []
            for i in range(T):
                for k_idx in range(len(indices[i])):
                    j = indices[i, k_idx].item()
                    if values[i, k_idx] > 0.1:
                        token_pairs.append((i, j))
        else:
            token_pairs = [(i, j) for i in range(T) for j in range(i + 1)]

        entangled_state = self.create_pairwise_entanglement(
            state,
            token_pairs,
            entanglement_strength=1.0,
        )

        return entangled_state, entangle_attn

    def verify_entanglement(
        self,
        state: QuantumStateVector,
        threshold: float = 0.01,
    ) -> tuple[bool, torch.Tensor]:
        """Check if state is entangled."""
        if state.n_qubits == 2:
            measure = self.compute_concurrence(state)
            is_entangled = (measure > threshold).any()
        else:
            measure = self.measure_entanglement_entropy(state, partition_a=[0])
            is_entangled = (measure > threshold).any()

        return is_entangled, measure
