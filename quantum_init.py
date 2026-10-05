"""Quantum Processor Initialization and Registry.

Entry point for initializing VirtualQuantumProcessor instances and quantum registries.
"""

from __future__ import annotations

from typing import Optional, Union, Any
import torch

try:
    from quantum.quantum_processor import VirtualQuantumProcessor
    from quantum.measurement import CollapseProtocol
except ImportError:
    from voidformer.quantum.quantum_processor import VirtualQuantumProcessor
    from voidformer.quantum.measurement import CollapseProtocol

# Global registry of initialized quantum processors
_PROCESSOR_REGISTRY: dict[str, VirtualQuantumProcessor] = {}


def initialize_quantum_processor(
    d_model: int = 256,
    n_qubits_per_token: int = 4,
    max_seq_len: int = 512,
    collapse_protocol: Union[str, CollapseProtocol] = CollapseProtocol.ENTROPY_GATED,
    enable_entanglement: bool = True,
    device: Optional[torch.device] = None,
    name: str = "default_processor",
) -> VirtualQuantumProcessor:
    """Initialize or retrieve a VirtualQuantumProcessor instance."""
    if isinstance(collapse_protocol, str):
        try:
            collapse_protocol = CollapseProtocol(collapse_protocol.lower())
        except ValueError:
            collapse_protocol = CollapseProtocol.ENTROPY_GATED

    processor = VirtualQuantumProcessor(
        d_model=d_model,
        n_qubits_per_token=n_qubits_per_token,
        max_seq_len=max_seq_len,
        collapse_protocol=collapse_protocol,
        enable_entanglement=enable_entanglement,
        device=device,
    )

    _PROCESSOR_REGISTRY[name] = processor
    return processor


def get_quantum_processor(name: str = "default_processor") -> Optional[VirtualQuantumProcessor]:
    """Retrieve an initialized processor from global registry."""
    return _PROCESSOR_REGISTRY.get(name)


def get_backend(kind: str = "simulator", **kwargs: Any) -> Any:
    """Factory function to get quantum backend."""
    kind = kind.lower()
    if kind == "simulator":
        return None
    try:
        from voidformer.quantum.ibm_backend import IBMBackend
    except ImportError:
        from quantum.ibm_backend import IBMBackend

    use_sim = kind in ("ibm_aer", "aer", "simulator_aer")
    return IBMBackend(use_simulator=use_sim, **kwargs)


if __name__ == "__main__":
    print("Initializing Virtual Quantum Processor...")
    proc = initialize_quantum_processor()
    print(f"Initialized processor with {proc.n_qubits} qubits per token, state dim={proc.state_dim}")
