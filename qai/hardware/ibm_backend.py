"""IBM Quantum QPU backend bridge using Qiskit Runtime and PennyLane-Qiskit."""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, List
import pennylane as qml
from qai.hardware.base_backend import BaseQuantumBackend
from qai.hardware.authentication import QuantumAuthenticator

logger = logging.getLogger(__name__)


class IBMBackend(BaseQuantumBackend):
    """
    Genuine IBM Quantum hardware backend bridge.
    Uses Qiskit Runtime Service and PennyLane-Qiskit's `qiskit.remote`.
    """

    def __init__(
        self,
        backend_name: str = "ibm_sherbrooke",
        allow_hardware_execution: bool = False,
        shots: int = 1024,
    ) -> None:
        super().__init__(name=backend_name, is_hardware=True, is_noisy=True)
        self.allow_hardware_execution = allow_hardware_execution
        self.shots = shots
        self._service = None
        self._target_backend = None

    def initialize_service(self) -> bool:
        """Initializes connection to Qiskit Runtime Service if credentials exist."""
        token = QuantumAuthenticator.get_ibm_token()
        instance = QuantumAuthenticator.get_ibm_instance()

        if not token:
            logger.warning("No IBM Quantum token found in environment. Hardware execution disabled.")
            return False

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self._service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
            try:
                self._target_backend = self._service.backend(self.name)
            except Exception:
                # If specific backend isn't available, find the least busy operational backend
                self._target_backend = self._service.least_busy(operational=True, simulator=False)
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with IBM Quantum: {e}")
            self._service = None
            self._target_backend = None
            return False

    def validate_requirements(self, n_qubits: int) -> bool:
        if not self.allow_hardware_execution:
            return False
        if not self._service:
            if not self.initialize_service():
                return False
        if self._target_backend is None:
            return False
        return self._target_backend.num_qubits >= n_qubits

    def get_device(self, n_qubits: int, shots: Optional[int] = None, **kwargs: Any) -> qml.devices.Device:
        if not self.allow_hardware_execution:
            raise PermissionError("Hardware execution is disabled. Set allow_hardware_execution=True to confirm.")

        if not self._target_backend and not self.initialize_service():
            raise RuntimeError("HARDWARE_NOT_VERIFIED: Unable to authenticate with IBM Quantum.")

        shots = shots or self.shots
        return qml.device(
            "qiskit.remote",
            wires=n_qubits,
            backend=self._target_backend,
            shots=shots,
        )

    def get_hardware_info(self) -> Dict[str, Any]:
        has_auth = QuantumAuthenticator.has_valid_credentials()
        return {
            "name": self.name,
            "type": "physical_qpu",
            "is_hardware": True,
            "is_noisy": True,
            "authenticated": has_auth,
            "execution_allowed": self.allow_hardware_execution,
            "status": "connected" if (self._target_backend is not None) else "disconnected",
            "num_qubits": getattr(self._target_backend, "num_qubits", None) if self._target_backend else None,
        }
