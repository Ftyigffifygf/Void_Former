"""Abstract base backend interface for quantum circuit execution."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import pennylane as qml


class BaseQuantumBackend(ABC):
    """Abstract base class for all simulator and hardware execution backends."""

    def __init__(self, name: str, is_hardware: bool = False, is_noisy: bool = False) -> None:
        self.name = name
        self.is_hardware = is_hardware
        self.is_noisy = is_noisy

    @abstractmethod
    def get_device(self, n_qubits: int, shots: Optional[int] = None, **kwargs: Any) -> qml.devices.Device:
        """Returns the PennyLane device configured for this backend."""
        pass

    @abstractmethod
    def validate_requirements(self, n_qubits: int) -> bool:
        """Validates qubit counts, connectivity, and hardware availability."""
        pass

    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """Returns hardware specifications, active status, and metadata."""
        pass
