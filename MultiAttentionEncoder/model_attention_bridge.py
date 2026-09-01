"""Compatibility import for the Attention Encoder -> Hawkes Memory bridge.

The former implementation targeted the removed TreeVAE package.  The active
downstream model is now ``Memory.LatentHawkesTree.HawkesTree``; keep this module
name as a stable import path while exposing the canonical adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path


_MEMORY_ROOT = Path(__file__).resolve().parents[1] / "Memory"
if str(_MEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMORY_ROOT))

from AttentionEncoderAdapter import AttentionMemoryEncoder, NodewiseEncoderOutput


__all__ = ["AttentionMemoryEncoder", "NodewiseEncoderOutput"]
