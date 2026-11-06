"""Core orchestration for the Weight Space Programming skill.

The :class:`WeightSpaceProgram` class discovers available adapters defined in
``weight_space_programming_skill.adapters`` and runs them sequentially. Each
adapter implements the :class:`BaseAdapter` interface with a ``run(prompt)``
method. The core returns a concatenated string of all adapter results. If no
adapters are available it returns a helpful placeholder.
"""

import importlib
import re
from typing import List

from weight_space_programming_skill.adapters import BaseAdapter


class WeightSpaceProgram:
    """Orchestrator that runs all registered adapters.

    The class lazily discovers adapter subclasses in the ``adapters`` package.
    ``run`` accepts an arbitrary *prompt* string, passes it to each adapter in
    order, and joins their outputs.
    """

    def __init__(self) -> None:
        self._adapters: List[BaseAdapter] = self._load_adapters()

    def _camel_to_snake(self, name: str) -> str:
        """Convert ``CamelCase`` class name to ``snake_case`` module name.
        Example: ``GenericAdapter`` -> ``generic_adapter``.
        """
        s1 = re.sub('(.)([A-Z][a-z]+)', r"\1_\2", name)
        return re.sub('([a-z0-9])([A-Z])', r"\1_\2", s1).lower()

    def _load_adapters(self) -> List[BaseAdapter]:
        """Import the ``adapters`` package and instantiate every concrete subclass.

        The ``adapters`` package defines ``__all__`` with the public adapter names.
        We import each name, instantiate it, and return the list. If the package
        cannot be imported or contains no adapters, an empty list is returned.
        """
        try:
            pkg = importlib.import_module("weight_space_programming_skill.adapters")
        except Exception:
            return []
        adapters: List[BaseAdapter] = []
        for name in getattr(pkg, "__all__", []):
            if name == "BaseAdapter":
                continue
            try:
                module_name = self._camel_to_snake(name)
                module = importlib.import_module(
                    f"weight_space_programming_skill.adapters.{module_name}"
                )
                adapter_cls = getattr(module, name)
                adapters.append(adapter_cls())
            except Exception:
                # Gracefully ignore failing adapters
                continue
        return adapters

    def run(self, prompt: str) -> str:
        """Execute *prompt* through all available adapters.

        Returns a single string. If no adapters are loaded, the result starts
        with ``[No adapters]`` to make the behaviour explicit for the test suite.
        """
        if not self._adapters:
            return "[No adapters]"
        results = []
        for adapter in self._adapters:
            try:
                results.append(adapter.run(prompt))
            except Exception as e:
                results.append(f"[adapter error: {e}]")
        return " ".join(results)

    __call__ = run
