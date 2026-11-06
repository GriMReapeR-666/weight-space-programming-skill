"""Adapter package exports.

Provides a minimal import pattern for the adapters used in tests.
"""

from .base import BaseAdapter
from .generic_adapter import GenericAdapter

__all__ = [
    "BaseAdapter",
    "GenericAdapter",
]
