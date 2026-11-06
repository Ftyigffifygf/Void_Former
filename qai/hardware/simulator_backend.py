"""Ideal quantum simulator backends (default.qubit, lightning.qubit, qiskit.aer)."""

from __future__ import annotations
from typing import Any, Dict, Optional
import pennylane as qml
from qai.hardware.base_backend import BaseQuantumBackend
from qai.core.device_manager import DeviceManager


class SimulatorBackend(BaseQuantumBackend):
    """Execution backend using local state-vector quantum simulators."""

    def __init__(self, backend_name: str = "default.qubit") -> None:
        super().__init__(name=backend_name, is_hardware=False, is_noisy=False)

    def get_device(self, n_qubits: int, shots: Optional[int] = None, **kwargs: Any) -> qml.devices.Device:
        return DeviceManager.create_device(
            backend_name=self.name,
            n_qubits=n_qubits,
            shots=shots,
            **kwargs,
        )

    def validate_requirements(self, n_qubits: int) -> bool:
        if n_qubits <= 0 or n_qubits > 28:
            return False
        return True

    def get_hardware_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "simulator",
            "is_hardware": False,
            "is_noisy": False,
            "status": "active",
            "max_qubits": 28,
        }
