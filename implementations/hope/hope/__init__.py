"""HOPE / Nested Learning (arXiv:2512.24695) — faithful PyTorch implementation."""

from .model import HOPE, HOPEConfig, HOPEBlock, SelfModifyingTitans, CMS

__all__ = ["HOPE", "HOPEConfig", "HOPEBlock", "SelfModifyingTitans", "CMS"]
__version__ = "1.0.0"
