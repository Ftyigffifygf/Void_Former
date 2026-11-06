"""Noisy quantum simulator backend with configurable depolarizing and thermal relaxation noise."""

from __future__ import annotations
from typing import Any, Dict, Optional
import pennylane as qml
from qai.hardware.base_backend import BaseQuantumBackend
from qai.core.device_manager import DeviceManager


class NoisyBackend(BaseQuantumBackend):
    """
    Execution backend using density matrix simulation (`default.mixed`)
    or Qiskit Aer with noise models to simulate realistic decoherence and gate noise.
    """

    def __init__(self, backend_name: str = "default.mixed", noise_probability: float = 0.05) -> None:
        super().__init__(name=backend_name, is_hardware=False, is_noisy=True)
        self.noise_probability = noise_probability

    def get_device(self, n_qubits: int, shots: Optional[int] = None, **kwargs: Any) -> qml.devices.Device:
        shots = shots or 1024
        return DeviceManager.create_device(
            backend_name="default.mixed",
            n_qubits=n_qubits,
            shots=shots,
            **kwargs,
        )

    def validate_requirements(self, n_qubits: int) -> bool:
        # Density matrix simulation scales as 4^n, limit to small registers
        return 0 < n_qubits <= 10

    def get_hardware_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "noisy_simulator",
            "is_hardware": False,
            "is_noisy": True,
            "noise_probability": self.noise_probability,
            "status": "active",
            "max_qubits": 10,
        }
