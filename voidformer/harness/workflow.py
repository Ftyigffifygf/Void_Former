"""Customizable Graph/DAG Workflow Engine for Harness Package."""

from __future__ import annotations

from typing import Dict, List, Any
from .node import WorkflowNode


class WorkflowDAG:
    """Directed Acyclic Graph workflow engine supporting drag-and-drop node graph execution."""

    def __init__(self, name: str = "VoidFormer_Team_Workflow"):
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[tuple[str, str]] = []

    def add_node(self, node: WorkflowNode):
        self.nodes[node.node_id] = node

    def add_edge(self, source_id: str, target_id: str):
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append((source_id, target_id))

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = initial_context or {}
        for node_id, node in self.nodes.items():
            context = node.execute(context)
        return context
