"""Real Quantum Backend Integration Interface.

Provides conversion tools to translate VoidFormer QuantumCircuit objects into Qiskit QuantumCircuit
objects and interface with Qiskit Aer simulators or IBM Quantum real hardware backends.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, List
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit_aer import AerSimulator

from .quantum_processor import QuantumCircuit


class BackendIntegration:
    """Interface for running VoidFormer quantum circuits on Qiskit Aer or real QPUs."""

    @staticmethod
    def to_qiskit_circuit(circuit: QuantumCircuit) -> QiskitCircuit:
        """Convert a VoidFormer QuantumCircuit into a native Qiskit QuantumCircuit.

        Args:
            circuit: VoidFormer QuantumCircuit object

        Returns:
            Qiskit QuantumCircuit object
        """
        qc = QiskitCircuit(circuit.n_qubits)

        for gate_item in circuit.gates:
            if len(gate_item) == 2:
                gate_name, targets = gate_item
                params = {}
            else:
                gate_name, targets, params = gate_item

            if gate_name == "H":
                qc.h(targets[0])
            elif gate_name in ("X", "Pauli-X"):
                qc.x(targets[0])
            elif gate_name in ("Y", "Pauli-Y"):
                qc.y(targets[0])
            elif gate_name in ("Z", "Pauli-Z"):
                qc.z(targets[0])
            elif gate_name.startswith("RX"):
                angle = params.get("angle", 0.0)
                qc.rx(angle, targets[0])
            elif gate_name.startswith("RY"):
                angle = params.get("angle", 0.0)
                qc.ry(angle, targets[0])
            elif gate_name.startswith("RZ"):
                angle = params.get("angle", 0.0)
                qc.rz(angle, targets[0])
            elif gate_name == "CNOT":
                qc.cx(targets[0], targets[1])
            elif gate_name == "Toffoli":
                qc.mcx([targets[0], targets[1]], targets[2])
            elif gate_name.startswith("Phase") or gate_name == "P":
                angle = params.get("angle", 0.0)
                qc.p(angle, targets[0])
            elif gate_name.startswith("CPhase") or gate_name == "CP":
                angle = params.get("angle", 0.0)
                qc.cp(angle, targets[0], targets[1])
            elif gate_name == "MEASURE":
                qc.measure_all()

        return qc

    @classmethod
    def execute_on_aer(
        cls,
        circuit: QuantumCircuit,
        shots: int = 1024,
    ) -> Dict[str, int]:
        """Execute a VoidFormer circuit on Qiskit Aer simulator."""
        qc = cls.to_qiskit_circuit(circuit)
        if not any(inst.operation.name == "measure" for inst in qc.data):
            qc.measure_all()

        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        return result.get_counts(qc)

    @classmethod
    def execute_batch_on_aer(
        cls,
        circuits: List[QuantumCircuit],
        shots: int = 1024,
    ) -> List[Dict[str, int]]:
        """Execute a batch of VoidFormer circuits on Qiskit Aer simulator."""
        qiskit_circuits = []
        for circuit in circuits:
            qc = cls.to_qiskit_circuit(circuit)
            if not any(inst.operation.name == "measure" for inst in qc.data):
                qc.measure_all()
            qiskit_circuits.append(qc)

        simulator = AerSimulator()
        job = simulator.run(qiskit_circuits, shots=shots)
        result = job.result()
        return [result.get_counts(qc) for qc in qiskit_circuits]

    @staticmethod
    def get_ibm_service(token: Optional[str] = None) -> Any:
        """Authenticate and retrieve IBM Quantum QiskitRuntimeService instance."""
        if not token:
            raise ValueError("IBM Quantum token must be provided")
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            return QiskitRuntimeService(channel="ibm_quantum", token=token)
        except Exception as e:
            raise RuntimeError(f"Failed to authenticate with IBM Quantum service: {e}")

    @staticmethod
    def poll_job(job: Any, timeout_seconds: float = 300.0, poll_interval: float = 0.5) -> Any:
        """Poll IBM Quantum job status until completion, error, or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            status = job.status()
            if status == "DONE":
                return job.result()
            elif status == "ERROR":
                raise RuntimeError("IBM Quantum job failed with ERROR status")
            time.sleep(poll_interval)
        if hasattr(job, "cancel"):
            job.cancel()
        raise TimeoutError(f"IBM Quantum job timed out after {timeout_seconds} seconds")
