"""Hardware validation status and reporting."""

from __future__ import annotations
from typing import Dict, Any, Optional
from qai.hardware.authentication import QuantumAuthenticator
from qai.hardware.ibm_backend import IBMBackend


class HardwareValidator:
    """Validates real QPU accessibility, authorization, and execution status."""

    @staticmethod
    def check_hardware_status(backend_name: str = "ibm_sherbrooke", allow_hardware: bool = False) -> Dict[str, Any]:
        """
        Performs an honest validation of hardware connectivity:
        Returns PASS, FAIL, or NOT AVAILABLE (HARDWARE_NOT_VERIFIED).
        """
        has_creds = QuantumAuthenticator.has_valid_credentials()
        if not has_creds:
            return {
                "hardware_access": "NOT AVAILABLE",
                "circuit_execution": "NOT AVAILABLE",
                "optimization_steps": "NOT AVAILABLE",
                "reason": "HARDWARE_NOT_VERIFIED: No IBM Quantum credentials found in environment (QISKIT_IBM_TOKEN / IBMQ_TOKEN).",
                "authenticated": False,
            }

        if not allow_hardware:
            return {
                "hardware_access": "AUTHENTICATED",
                "circuit_execution": "NOT AUTHORIZED",
                "optimization_steps": "NOT AUTHORIZED",
                "reason": "Hardware execution disabled by user safeguard. Enable allow_hardware_execution to proceed.",
                "authenticated": True,
            }

        ibm_backend = IBMBackend(backend_name=backend_name, allow_hardware_execution=True)
        connected = ibm_backend.initialize_service()

        if not connected:
            return {
                "hardware_access": "FAIL",
                "circuit_execution": "FAIL",
                "optimization_steps": "FAIL",
                "reason": "Failed to connect to IBM Quantum Runtime Service.",
                "authenticated": True,
            }

        return {
            "hardware_access": "PASS",
            "circuit_execution": "READY",
            "optimization_steps": "READY",
            "reason": f"Successfully connected to IBM Quantum backend {ibm_backend._target_backend.name}",
            "authenticated": True,
            "backend_name": ibm_backend._target_backend.name,
            "num_qubits": ibm_backend._target_backend.num_qubits,
        }
