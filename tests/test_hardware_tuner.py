"""Unit tests for System Hardware Resource Auto-Tuner."""

from __future__ import annotations

from voidformer.quantum.hardware_tuner import QuantumHardwareResourceTuner
from voidformer.harness.quantum_bridge import QuantumEngineeringBridge


def test_system_specs_inspection():
    specs = QuantumHardwareResourceTuner.inspect_system_specs()

    assert "cpu_cores" in specs
    assert specs["cpu_cores"] >= 1
    assert "total_ram_gb" in specs
    assert "available_ram_gb" in specs
    assert "has_cuda" in specs


def test_compute_optimal_simulation_config():
    tuner = QuantumHardwareResourceTuner(target_memory_fraction=0.5)
    config = tuner.compute_optimal_simulation_config()

    assert "selected_device" in config
    assert config["selected_device"] in ["cpu", "cuda"]
    assert "optimal_num_threads" in config
    assert config["optimal_num_threads"] >= 1
    assert "max_safe_simulated_qubits" in config
    assert config["max_safe_simulated_qubits"] >= 2


def test_quantum_engineering_bridge_hardware_autotune():
    bridge = QuantumEngineeringBridge(backend_type="auto")
    config = bridge.auto_tune_hardware_resources()

    assert "system_specs" in config
    assert "max_safe_simulated_qubits" in config
    assert config["max_safe_simulated_qubits"] >= 2
