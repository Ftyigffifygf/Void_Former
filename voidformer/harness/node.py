"""Workflow Node Definitions for Customizable Harness Engine."""

from __future__ import annotations

from typing import Dict, Any, List, Optional


class WorkflowNode:
    """Base Node for n8n-style graph workflow engine."""

    def __init__(self, node_id: str, name: str, node_type: str, config: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.config = config or {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node computation on workflow context."""
        context[self.node_id] = {"status": "executed", "name": self.name}
        return context


class AgentNode(WorkflowNode):
    """Autonomous agent node with configurable loop count and task skills."""

    def __init__(self, node_id: str, name: str, skill: str, max_loops: int = 25, config: Optional[Dict[str, Any]] = None):
        super().__init__(node_id, name, "agent", config)
        self.skill = skill
        self.max_loops = max_loops

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        loop_results = []
        for loop in range(self.max_loops):
            result = f"Loop {loop+1}/{self.max_loops}: Executed skill '{self.skill}'"
            loop_results.append(result)

        context[self.node_id] = {
            "skill": self.skill,
            "loops_completed": self.max_loops,
            "loop_results": loop_results,
        }
        return context


class QuantumBackendNode(WorkflowNode):
    """Node connecting VoidFormer quantum simulation or QPU backends."""

    def __init__(self, node_id: str, name: str, backend_type: str = "statevector", config: Optional[Dict[str, Any]] = None):
        super().__init__(node_id, name, "quantum_backend", config)
        self.backend_type = backend_type

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context[self.node_id] = {
            "backend": self.backend_type,
            "qpu_bridge_active": True,
            "status": "connected",
        }
        return context
