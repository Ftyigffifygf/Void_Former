"""Authentication and credential management for IBM Quantum."""

from __future__ import annotations
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class QuantumAuthenticator:
    """Safely retrieves credentials from environment variables without logging or exposing secrets."""

    ENV_TOKEN_KEYS = ["QISKIT_IBM_TOKEN", "IBMQ_TOKEN", "IBM_QUANTUM_TOKEN"]
    ENV_INSTANCE_KEYS = ["QISKIT_IBM_INSTANCE", "IBMQ_INSTANCE"]

    @classmethod
    def get_ibm_token(cls) -> Optional[str]:
        for key in cls.ENV_TOKEN_KEYS:
            token = os.environ.get(key)
            if token and token.strip():
                return token.strip()
        return None

    @classmethod
    def get_ibm_instance(cls) -> Optional[str]:
        for key in cls.ENV_INSTANCE_KEYS:
            instance = os.environ.get(key)
            if instance and instance.strip():
                return instance.strip()
        return None

    @classmethod
    def has_valid_credentials(cls) -> bool:
        return cls.get_ibm_token() is not None
