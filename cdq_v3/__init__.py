"""
cdq_v3
=======

Implementation of the curiosity-optimization method described by the user.
This package wires together data loading, retrieval, OpenAI-backed LLM roles,
curiosity index computation, and mirror-descent inference-time learning.
"""

from .config import MethodConfig
from .method import CuriosityOptimizer

__all__ = ["MethodConfig", "CuriosityOptimizer"]
