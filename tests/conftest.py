"""Pytest configuration and backward-compatible namespace aliasing."""

import sys
import adapters
import core
import engine
import utils

# Register backwards-compatible namespace aliases for weight_space_programming_skill
mod = sys.modules.get("src") or type(sys)("weight_space_programming_skill")
mod.WeightSpaceProgram = core.WeightSpaceProgram  # type: ignore
sys.modules["weight_space_programming_skill"] = mod
sys.modules["weight_space_programming_skill.engine"] = engine
sys.modules["weight_space_programming_skill.adapters"] = adapters
sys.modules["weight_space_programming_skill.utils"] = utils
sys.modules["weight_space_programming_skill.core"] = core
