"""Backend registry for registering and discovering execution environments."""

from __future__ import annotations
from typing import Callable, Dict, Any, List
import pennylane as qml


class BackendRegistry:
    """Registry for quantum execution backends."""
    _backends: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, description: str, is_hardware: bool = False, is_noisy: bool = False) -> None:
        cls._backends[name.lower()] = {
            "name": name,
            "description": description,
            "is_hardware": is_hardware,
            "is_noisy": is_noisy,
        }

    @classmethod
    def list_backends(cls) -> List[Dict[str, Any]]:
        return list(cls._backends.values())

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        return cls._backends.get(name.lower(), {"name": name, "description": "Unknown backend", "is_hardware": False, "is_noisy": False})


# Register standard default backends
BackendRegistry.register("default.qubit", "PennyLane pure Python full statevector simulator")
BackendRegistry.register("lightning.qubit", "PennyLane high-performance C++ statevector simulator")
BackendRegistry.register("default.mixed", "PennyLane density matrix simulator for open quantum systems and noise")
BackendRegistry.register("qiskit.aer", "Qiskit Aer high-performance local simulator")
BackendRegistry.register("qiskit.remote", "IBM Quantum physical QPU execution via Qiskit Runtime", is_hardware=True)
