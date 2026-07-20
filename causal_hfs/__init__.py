"""
causal_hfs
==========

Reference implementation of the **Causality-Aware Stable Hierarchical Feature
Selection Framework** (Markov Blanket discovery + hybrid causal-statistical
clustering + consensus-based stability selection + Human-in-the-Loop).

The seven modular pipeline stages described in the paper map onto modules:

1. Data Preprocessing         -> ``preprocessing``
2. Causal Discovery           -> ``causal``
3. Structural Mapping         -> ``graph``
4. Hybrid Distance            -> ``distance``
5. Hierarchical Agglomeration -> ``clustering``
6. Representative Extraction  -> ``clustering`` (prototype selection)
7. Robustness Validation      -> ``consensus``

The orchestrator that wires them together lives in ``framework.CausalHFS``.
"""

from .framework import CausalHFS
from .config import FrameworkConfig

__all__ = ["CausalHFS", "FrameworkConfig"]
__version__ = "1.0.0"
