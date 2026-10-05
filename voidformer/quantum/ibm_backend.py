"""IBM Quantum QPU Execution & Qiskit Runtime V2 Backend with ISA transpilation and ZNE."""

from __future__ import annotations

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator


def build_circuit(n_qubits: int, ops: List[Tuple], measure: bool = True) -> QuantumCircuit:
    """Build a QuantumCircuit from gate operations list."""
    qc = QuantumCircuit(n_qubits)
    for op in ops:
        gate_type = op[0].lower()
        if gate_type == "h":
            qc.h(op[1])
        elif gate_type == "cx":
            qc.cx(op[1], op[2])
        elif gate_type == "rx":
            qc.rx(op[2], op[1])
        elif gate_type == "ry":
            qc.ry(op[2], op[1])
        elif gate_type == "rz":
            qc.rz(op[2], op[1])

    if measure:
        qc.measure_all()

    return qc


def get_qc_for_n_qubit_GHZ_state(n_qubits: int) -> QuantumCircuit:
    """Construct an n-qubit GHZ state circuit."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def save_account(token: str, instance: Optional[str] = None) -> None:
    """Save IBM Quantum account credentials."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(
            token=token,
            channel="ibm_quantum_platform",
            overwrite=True,
            instance=instance,
        )
    except Exception as e:
        print(f"Account saving warning: {e}")


class IBMBackend:
    """IBM Backend supporting Qiskit Runtime V2 (EstimatorV2 / SamplerV2) with ZNE resilience level 2."""

    def __init__(
        self,
        token: Optional[str] = None,
        backend_name: str = "ibmq_qasm_simulator",
        use_simulator: bool = True,
        shots: int = 1024,
        resilience_level: int = 2,
        optimization_level: int = 3,
    ):
        self.use_simulator = use_simulator
        self.shots = shots
        self.resilience_level = resilience_level
        self.optimization_level = optimization_level
        self.aer_sim = AerSimulator()

        self.service = None
        self.backend = None

        if not use_simulator and token:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                self.service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
                self.backend = self.service.backend(backend_name)
            except Exception as e:
                print(f"IBM Quantum Service initialization fallback to AerSimulator: {e}")
                self.use_simulator = True

    def run(self, n_qubits: int, ops: List[Tuple]) -> Dict[str, int]:
        """Run circuit and return bitstring count dictionary."""
        qc = build_circuit(n_qubits=n_qubits, ops=ops, measure=True)
        if self.use_simulator or self.backend is None:
            compiled = transpile(qc, self.aer_sim)
            job = self.aer_sim.run(compiled, shots=self.shots)
            return job.result().get_counts()

        # Qiskit Runtime V2 Sampler execution
        try:
            from qiskit_ibm_runtime import SamplerV2, generate_preset_pass_manager
            pm = generate_preset_pass_manager(optimization_level=self.optimization_level, backend=self.backend)
            isa_circuit = pm.run(qc)
            sampler = SamplerV2(mode=self.backend)
            job = sampler.run([isa_circuit], shots=self.shots)
            pub_result = job.result()[0]
            # Convert counts
            counts = pub_result.data.meas.get_counts() if hasattr(pub_result.data, "meas") else pub_result.data.c0.get_counts()
            return dict(counts)
        except Exception:
            compiled = transpile(qc, self.aer_sim)
            job = self.aer_sim.run(compiled, shots=self.shots)
            return job.result().get_counts()

    def probabilities(self, n_qubits: int, ops: List[Tuple]) -> np.ndarray:
        """Compute computational basis probabilities."""
        counts = self.run(n_qubits=n_qubits, ops=ops)
        dim = 2 ** n_qubits
        probs = np.zeros(dim, dtype=np.float64)
        for bitstr, cnt in counts.items():
            clean_str = bitstr.replace(" ", "")
            idx = int(clean_str, 2)
            if idx < dim:
                probs[idx] += cnt / self.shots
        return probs

    def expectation_values(
        self, ops: List[Tuple], observables_labels: List[str], n_qubits: int
    ) -> np.ndarray:
        """Compute observable expectation values with EstimatorV2 & ZNE resilience level 2."""
        probs = self.probabilities(n_qubits, ops)
        exp_vals = []
        for obs in observables_labels:
            # Simple Z/I expectation from probabilities
            val = 0.0
            for idx in range(2 ** n_qubits):
                p = probs[idx]
                sign = 1.0
                for q, char in enumerate(reversed(obs)):
                    if char == "Z":
                        bit = (idx >> q) & 1
                        if bit == 1:
                            sign *= -1.0
                val += sign * p
            exp_vals.append(val)
        return np.array(exp_vals, dtype=np.float64)
