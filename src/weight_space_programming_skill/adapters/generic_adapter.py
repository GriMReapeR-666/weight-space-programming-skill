"""Generic adapter stub.

Simply echoes the prompt back. Replace with real implementation when needed.
"""

from .base import BaseAdapter
from ..utils import logger


class GenericAdapter(BaseAdapter):
    def run(self, prompt: str) -> str:
        """Echo the prompt with a generic tag.

        Args:
            prompt: The input prompt string.
        Returns:
            A string prefixed with ``[generic]``.
        """
        logger.info("GenericAdapter received prompt: %s", prompt)
        return f"[generic] {prompt}"


# Export for discovery
__all__ = ["GenericAdapter"]
