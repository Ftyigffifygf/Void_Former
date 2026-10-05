"""Quantum Gate primitives for VQC simulation."""

from __future__ import annotations

import math
import torch


class QuantumGate:
    def verify_unitary() -> bool:
        return True

    def apply(self, state, target_qubits):
        raise NotImplementedError


class HadamardGate(QuantumGate):
    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q = target_qubits[0]
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step = 2 ** q

        m = 1.0 / math.sqrt(2.0)
        m00, m01, m10, m11 = m, m, m, -m

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx0 = torch.where((i & step) == 0)[0]
        idx1 = idx0 + step

        v0 = amps[..., idx0]
        v1 = amps[..., idx1]

        v0_new = m00 * v0 + m01 * v1
        v1_new = m10 * v0 + m11 * v1

        out_amps = amps.clone()
        out_amps[..., idx0] = v0_new
        out_amps[..., idx1] = v1_new
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class PauliXGate(QuantumGate):
    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q = target_qubits[0]
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step = 2 ** q

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx0 = torch.where((i & step) == 0)[0]
        idx1 = idx0 + step

        out_amps = amps.clone()
        out_amps[..., idx0] = amps[..., idx1]
        out_amps[..., idx1] = amps[..., idx0]
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class PauliYGate(QuantumGate):
    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q = target_qubits[0]
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step = 2 ** q

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx0 = torch.where((i & step) == 0)[0]
        idx1 = idx0 + step

        out_amps = amps.clone()
        out_amps[..., idx0] = -1j * amps[..., idx1]
        out_amps[..., idx1] = 1j * amps[..., idx0]
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class PauliZGate(QuantumGate):
    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q = target_qubits[0]
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step = 2 ** q

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx1 = torch.where((i & step) != 0)[0]

        out_amps = amps.clone()
        out_amps[..., idx1] = -amps[..., idx1]
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class PhaseGate(QuantumGate):
    def __init__(self, phi: float = math.pi / 2.0):
        self.phi = phi

    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q = target_qubits[0]
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step = 2 ** q

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx1 = torch.where((i & step) != 0)[0]

        out_amps = amps.clone()
        out_amps[..., idx1] = amps[..., idx1] * torch.exp(1j * torch.tensor(self.phi, device=amps.device))
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class SwapGate(QuantumGate):
    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q1, q2 = target_qubits
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step1 = 2 ** q1
        step2 = 2 ** q2

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx01 = torch.where(((i & step1) == 0) & ((i & step2) != 0))[0]
        idx10 = idx01 + step1 - step2

        out_amps = amps.clone()
        out_amps[..., idx01] = amps[..., idx10]
        out_amps[..., idx10] = amps[..., idx01]
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class ToffoliGate(QuantumGate):
    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        c1, c2, t = target_qubits
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step_c1 = 2 ** c1
        step_c2 = 2 ** c2
        step_t = 2 ** t

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx0 = torch.where(((i & step_c1) != 0) & ((i & step_c2) != 0) & ((i & step_t) == 0))[0]
        idx1 = idx0 + step_t

        out_amps = amps.clone()
        out_amps[..., idx0] = amps[..., idx1]
        out_amps[..., idx1] = amps[..., idx0]
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class RXGate(QuantumGate):
    def __init__(self, theta: torch.Tensor | float):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        self.theta = theta

    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q = target_qubits[0]
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step = 2 ** q

        theta = self.theta
        cos = torch.cos(theta / 2.0)
        sin = torch.sin(theta / 2.0)

        m00 = cos
        m01 = -1j * sin
        m10 = -1j * sin
        m11 = cos

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx0 = torch.where((i & step) == 0)[0]
        idx1 = idx0 + step

        v0 = amps[..., idx0]
        v1 = amps[..., idx1]

        if theta.ndim > 0 and theta.shape[0] == amps.shape[0]:
            m00 = m00.unsqueeze(-1)
            m01 = m01.unsqueeze(-1)
            m10 = m10.unsqueeze(-1)
            m11 = m11.unsqueeze(-1)

        v0_new = m00 * v0 + m01 * v1
        v1_new = m10 * v0 + m11 * v1

        out_amps = amps.clone()
        out_amps[..., idx0] = v0_new
        out_amps[..., idx1] = v1_new
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class RYGate(QuantumGate):
    def __init__(self, theta: torch.Tensor | float):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        self.theta = theta

    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q = target_qubits[0]
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step = 2 ** q

        theta = self.theta
        cos = torch.cos(theta / 2.0)
        sin = torch.sin(theta / 2.0)

        m00 = cos
        m01 = -sin
        m10 = sin
        m11 = cos

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx0 = torch.where((i & step) == 0)[0]
        idx1 = idx0 + step

        v0 = amps[..., idx0]
        v1 = amps[..., idx1]

        if theta.ndim > 0 and theta.shape[0] == amps.shape[0]:
            m00 = m00.unsqueeze(-1)
            m01 = m01.unsqueeze(-1)
            m10 = m10.unsqueeze(-1)
            m11 = m11.unsqueeze(-1)

        v0_new = m00 * v0 + m01 * v1
        v1_new = m10 * v0 + m11 * v1

        out_amps = amps.clone()
        out_amps[..., idx0] = v0_new
        out_amps[..., idx1] = v1_new
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class RZGate(QuantumGate):
    def __init__(self, theta: torch.Tensor | float):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        self.theta = theta

    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        q = target_qubits[0]
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits
        step = 2 ** q

        theta = self.theta
        m00 = torch.exp(-1j * theta / 2.0)
        m11 = torch.exp(1j * theta / 2.0)

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)
        idx0 = torch.where((i & step) == 0)[0]
        idx1 = idx0 + step

        v0 = amps[..., idx0]
        v1 = amps[..., idx1]

        if theta.ndim > 0 and theta.shape[0] == amps.shape[0]:
            m00 = m00.unsqueeze(-1)
            m11 = m11.unsqueeze(-1)

        v0_new = m00 * v0
        v1_new = m11 * v1

        out_amps = amps.clone()
        out_amps[..., idx0] = v0_new
        out_amps[..., idx1] = v1_new
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)


class CNOTGate(QuantumGate):
    def verify_unitary(self) -> bool:
        return True

    def apply(self, state, target_qubits):
        ctrl, target = target_qubits
        n_qubits = state.n_qubits
        dim = 2 ** n_qubits

        ctrl_step = 2 ** ctrl
        target_step = 2 ** target

        amps = state.amplitudes
        i = torch.arange(dim, device=amps.device)

        idx_ctrl1 = (i & ctrl_step) != 0
        idx_target0 = (i & target_step) == 0
        idx0 = torch.where(idx_ctrl1 & idx_target0)[0]
        idx1 = idx0 + target_step

        v0 = amps[..., idx0]
        v1 = amps[..., idx1]

        out_amps = amps.clone()
        out_amps[..., idx0] = v1
        out_amps[..., idx1] = v0
        return type(state)(amplitudes=out_amps, n_qubits=n_qubits)
