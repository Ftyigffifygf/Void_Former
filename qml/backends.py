"""Backend configuration for PennyLane devices (Simulator / Aer / IBM Hardware)."""

from __future__ import annotations

import os
from typing import Optional, Any
import pennylane as qml


def get_pennylane_device(
    backend_kind: str = "simulator",
    n_qubits: int = 4,
    shots: Optional[int] = None,
    backend_name: Optional[str] = None,
) -> qml.Device:
    """Create a PennyLane device for hybrid layer processing.

    Args:
        backend_kind: 'simulator' (default.qubit), 'ibm_aer', or 'ibm_hardware'
        n_qubits: Number of qubits in circuit
        shots: Number of measurement shots (None for exact statevector)
        backend_name: Specific IBM backend name if using Qiskit/IBM

    Returns:
        PennyLane Device object
    """
    kind = backend_kind.lower()
    if kind in ("simulator", "default.qubit", "cpu"):
        return qml.device("default.qubit", wires=n_qubits, shots=shots)

    if kind in ("ibm_aer", "aer"):
        try:
            return qml.device("qiskit.aer", wires=n_qubits, shots=shots or 1024)
        except Exception:
            # Fallback to default.qubit if pennylane-qiskit plugin isn't present
            return qml.device("default.qubit", wires=n_qubits, shots=shots or 1024)

    if kind in ("ibm_hardware", "ibm", "qiskit.ibmq"):
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            print("[QML Backend Warning] No IBM_QUANTUM_TOKEN set. Falling back to noisy Aer simulator/fake backend.")
            try:
                from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
                fake_backend = FakeSherbrooke()
                return qml.device(
                    "qiskit.aer",
                    wires=n_qubits,
                    shots=shots or 1024,
                    backend=fake_backend,
                )
            except Exception:
                return qml.device("default.qubit", wires=n_qubits, shots=shots or 1024)

        try:
            return qml.device(
                "qiskit.ibmq",
                wires=n_qubits,
                backend=backend_name or "ibm_kyiv",
                ibmqx_token=token,
                shots=shots or 1024,
            )
        except Exception as e:
            print(f"[QML Backend Error] Failed to connect to IBM Hardware ({e}). Falling back to Aer simulator.")
            return qml.device("default.qubit", wires=n_qubits, shots=shots or 1024)

    raise ValueError(f"Unknown backend_kind: {backend_kind}")
