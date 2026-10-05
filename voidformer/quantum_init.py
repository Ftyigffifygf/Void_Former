"""Quantum Backend initialization factory for Voidformer."""

from __future__ import annotations

from typing import Optional, Any
from voidformer.quantum.ibm_backend import IBMBackend


def get_backend(backend_type: str = "simulator") -> Optional[Any]:
    if backend_type == "simulator":
        return None
    elif backend_type in ("ibm_aer", "ibm"):
        return IBMBackend(use_simulator=True)
    return None
