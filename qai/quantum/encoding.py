"""Quantum data encoding techniques for mapping classical features into quantum states."""

from __future__ import annotations
import math
import torch
import pennylane as qml
from typing import Sequence, Union


class QuantumEncoder:
    """Provides methods for encoding classical features into quantum circuits."""

    @staticmethod
    def angle_embedding(features: Sequence[Union[float, torch.Tensor]], wires: Sequence[int], rotation: str = "Y") -> None:
        """Encodes features as single-qubit rotations."""
        rot = rotation.upper()
        for i, wire in enumerate(wires):
            val = features[i % len(features)]
            if rot == "X":
                qml.RX(val, wires=wire)
            elif rot == "Z":
                qml.RZ(val, wires=wire)
            else:
                qml.RY(val, wires=wire)

    @staticmethod
    def iqp_embedding(features: Sequence[Union[float, torch.Tensor]], wires: Sequence[int], pattern: str = "ring") -> None:
        """
        Instantaneous Quantum Polynomial (IQP) style feature map:
        Hadamard on all wires -> RZ(feature) -> RZZ(feature_i * feature_j) entangling phase gates.
        """
        for wire in wires:
            qml.Hadamard(wires=wire)

        n = len(wires)
        for i, wire in enumerate(wires):
            qml.RZ(features[i % len(features)], wires=wire)

        if n > 1:
            for i in range(n):
                j = (i + 1) % n if pattern == "ring" else i + 1
                if j < n or pattern == "ring":
                    prod = features[i % len(features)] * features[j % len(features)]
                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.RZ(prod, wires=wires[j])
                    qml.CNOT(wires=[wires[i], wires[j]])

    @staticmethod
    def amplitude_embedding(features: Sequence[Union[float, torch.Tensor]], wires: Sequence[int]) -> None:
        """Encodes features into the statevector amplitudes (requires 2^n normalized features)."""
        # Ensure normalized features
        qml.AmplitudeEmbedding(features=features, wires=wires, normalize=True, pad_with=0.0)

    @staticmethod
    def data_reuploading_step(features: Sequence[Union[float, torch.Tensor]], weights: Sequence[Union[float, torch.Tensor]], wires: Sequence[int]) -> None:
        """
        Performs one data re-uploading step:
        Rotations combining features and variational weights: RZ(w1)*RY(f + w2)*RZ(w3)
        """
        for i, wire in enumerate(wires):
            f_val = features[i % len(features)]
            w0 = weights[i, 0] if len(weights.shape) > 1 else weights[0]
            w1 = weights[i, 1] if len(weights.shape) > 1 else weights[1]
            w2 = weights[i, 2] if len(weights.shape) > 1 else weights[2]
            qml.RZ(w0, wires=wire)
            qml.RY(f_val + w1, wires=wire)
            qml.RZ(w2, wires=wire)
