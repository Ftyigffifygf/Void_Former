"""Trainability and Barren Plateau empirical research suite."""

from __future__ import annotations
import torch
import numpy as np
from typing import List, Dict, Any
import pennylane as qml

from qai.core.device_manager import DeviceManager
from qai.quantum.encoding import QuantumEncoder
from qai.quantum.ansatz import VariationalAnsatz


def analyze_trainability_and_barren_plateaus(
    qubit_range: List[int] = [2, 4, 6],
    depth_range: List[int] = [1, 2, 4],
    n_samples: int = 10,
    init_strategy: str = "identity_near",  # 'identity_near' or 'random'
) -> List[Dict[str, Any]]:
    """
    Empirically investigates barren plateaus and gradient magnitudes/variances
    as a function of qubit register width and circuit depth.
    """
    results = []

    for n_qubits in qubit_range:
        dev = DeviceManager.create_device("default.qubit", n_qubits=n_qubits)
        wires = list(range(n_qubits))

        for n_layers in depth_range:
            gradients = []

            for _ in range(n_samples):
                if init_strategy == "identity_near":
                    weights = (torch.randn(n_layers, n_qubits, 2) * 0.05).requires_grad_(True)
                else:
                    weights = ((torch.rand(n_layers, n_qubits, 2) * 2.0 - 1.0) * np.pi).requires_grad_(True)

                inputs = torch.rand(n_qubits) * np.pi

                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def circuit(x_val, w_val):
                    QuantumEncoder.angle_embedding(x_val, wires)
                    VariationalAnsatz.hardware_efficient(w_val, wires, entanglement="ring")
                    return qml.expval(qml.PauliZ(0))

                out = circuit(inputs, weights)
                out.backward()

                if weights.grad is not None:
                    g = weights.grad.flatten().detach().cpu().numpy()
                    gradients.append(g)

            if gradients:
                all_g = np.concatenate(gradients)
                mean_norm = float(np.mean(np.abs(all_g)))
                variance = float(np.var(all_g))
            else:
                mean_norm = 0.0
                variance = 0.0

            results.append({
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "init_strategy": init_strategy,
                "mean_gradient_magnitude": mean_norm,
                "gradient_variance": variance,
                "is_trainable": mean_norm > 1e-4,
            })

    return results
