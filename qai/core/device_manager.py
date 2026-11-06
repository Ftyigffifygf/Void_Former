"""Device and backend management for PennyLane and quantum simulator/hardware."""

from __future__ import annotations
import logging
from typing import Any, Optional, Tuple
import pennylane as qml

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages creation, validation, and lifecycle of PennyLane devices."""

    SUPPORTED_BACKENDS = [
        "default.qubit",
        "lightning.qubit",
        "default.mixed",
        "qiskit.aer",
        "qiskit.remote",
    ]

    @staticmethod
    def create_device(
        backend_name: str = "default.qubit",
        n_qubits: int = 4,
        shots: Optional[int] = None,
        **kwargs: Any,
    ) -> qml.devices.Device:
        """Create a PennyLane device with error handling and parameter validation."""
        if n_qubits <= 0:
            raise ValueError(f"Number of qubits must be positive, got {n_qubits}")

        backend = backend_name.lower().strip()

        if backend == "default.qubit":
            return qml.device("default.qubit", wires=n_qubits, shots=shots)

        elif backend == "lightning.qubit":
            try:
                return qml.device("lightning.qubit", wires=n_qubits, shots=shots)
            except Exception as e:
                logger.warning(f"Failed to load lightning.qubit ({e}), falling back to default.qubit")
                return qml.device("default.qubit", wires=n_qubits, shots=shots)

        elif backend == "default.mixed":
            return qml.device("default.mixed", wires=n_qubits, shots=shots)

        elif backend == "qiskit.aer":
            try:
                return qml.device("qiskit.aer", wires=n_qubits, shots=shots or 1024, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize qiskit.aer device: {e}") from e

        elif backend == "qiskit.remote":
            backend_target = kwargs.get("backend_target") or kwargs.get("backend")
            if not backend_target:
                raise ValueError("qiskit.remote requires a valid Qiskit target backend or backend object")
            try:
                return qml.device("qiskit.remote", wires=n_qubits, backend=backend_target, shots=shots or 1024)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize qiskit.remote device: {e}") from e

        else:
            raise ValueError(f"Unsupported quantum backend '{backend_name}'. Supported: {DeviceManager.SUPPORTED_BACKENDS}")
