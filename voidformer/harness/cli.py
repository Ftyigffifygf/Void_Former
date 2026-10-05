"""Customizable Workflow CLI Dispatcher for Harness Package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from voidformer.harness.node import AgentNode, QuantumBackendNode
from voidformer.harness.workflow import WorkflowDAG
from voidformer.harness.ab_benchmark import run_ab_benchmark
from voidformer.harness.ablation_sweep import run_ablation_sweep
from voidformer.harness.scaling_profile import profile_scaling
from voidformer.harness.deepseek_quantum_harness import DeepSeekQuantumHarness
from voidformer.harness.model_factory import create_model
import torch


def run_custom_workflow():
    print("=" * 65)
    print("       N8N-STYLE VOIDFORMER TEAM WORKFLOW ENGINE RUNNER         ")
    print("=" * 65)

    dag = WorkflowDAG(name="Quantum_AI_Consensus_Team")

    # Add customizable agent nodes with high loop counts
    dag.add_node(AgentNode("node_1", "Quantum Algorithm Specialist", skill="Grover_Search", max_loops=10))
    dag.add_node(AgentNode("node_2", "Circuit Synthesis Agent", skill="CNOT_Entanglement", max_loops=15))
    dag.add_node(QuantumBackendNode("node_3", "Qiskit Aer QPU Bridge", backend_type="aer"))

    dag.add_edge("node_1", "node_2")
    dag.add_edge("node_2", "node_3")

    results = dag.run()
    for n_id, data in results.items():
        print(f"Node '{n_id}': {data}")

    return results


def run_deepseek_quantum_eval(steps: int = 10):
    print("=" * 65)
    print("   DEEPSEEK QUANTUM REASONING & GRPO REWARD HARNESS EVAL   ")
    print("=" * 65)

    model = create_model(model_type="quantum", vocab_size=100, d_model=64)
    harness = DeepSeekQuantumHarness(d_model=64, n_vqc_qubits=4, group_size=4)

    dummy_ids = torch.randint(0, 100, (2, 8))
    logits, diag = harness.evaluate_reasoning_task(model, dummy_ids)

    print(f"Harness: {diag['harness_name']}")
    print(f"Group Size: {diag['group_size']}")
    print(f"Group Rewards: {diag['group_rewards']}")
    print(f"Mean Group Reward: {diag['mean_group_reward']:.4f}")
    print(f"Best Trajectory Index: {diag['selected_best_trajectory_idx']}")
    return diag


def main():
    parser = argparse.ArgumentParser(description="VoidFormer Evaluation Harness & Team Workflow CLI")
    parser.add_argument("mode", choices=["workflow", "benchmark", "sweep", "profile", "deepseek-eval", "all"], help="Execution mode")
    parser.add_argument("--steps", type=int, default=10, help="Training steps")
    args = parser.parse_args()

    if args.mode in ["workflow", "all"]:
        run_custom_workflow()
    if args.mode in ["benchmark", "all"]:
        run_ab_benchmark(steps=args.steps)
    if args.mode in ["sweep", "all"]:
        run_ablation_sweep(steps=args.steps)
    if args.mode in ["profile", "all"]:
        profile_scaling()
    if args.mode in ["deepseek-eval", "all"]:
        run_deepseek_quantum_eval(steps=args.steps)


if __name__ == "__main__":
    main()
