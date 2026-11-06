"""Weight-Space Programming Engine (WSP v5.0).

Exposes the complete programmatic tool interface for inspecting, diagnosing,
intervening, optimizing, and verifying neural network weights.
"""

from .ablation_sandbox import run_activation_ablation, run_parameter_ablation
from .activation_engine import ActivationTracer, capture_activations, trace_generation
from .circuit_mapper import build_circuit_map
from .evaluation_engine import compare_models, evaluate_metrics
from .graph_engine import identify_architecture, reconstruct_graph
from .loader import ModelLoader
from .mcp_server import MCPServer
from .patch_engine import apply_patch, generate_patch, rollback_patch, validate_patch
from .search_engine import evaluate_candidate, optimize_patch
from .tensor_engine import (
    analyze_weight_directions,
    inspect_model,
    inspect_tensor,
    summarize_weights,
)
from .pareto_surgery_planner import ParetoSurgeryPlanner
from .subspace_feature_analyzer import (
    compute_spectral_regularization_delta,
    profile_matrix_subspace,
    profile_model_subspaces,
    project_delta_onto_null_space,
)
from .types import (
    AblationMode,
    AblationResult,
    ActivationTrace,
    CapabilitySubspaceMap,
    CircuitMap,
    CircuitNode,
    ComputationGraph,
    EvaluationResult,
    GraphNode,
    ModelFormat,
    ModelMetadata,
    PatchManifest,
    PatchOperation,
    SubspaceProfile,
    SurgeryAction,
    SurgeryPlan,
    TensorRole,
    TensorSummary,
)

__all__ = [
    # 17 Core Agent Tools
    "inspect_model",
    "identify_architecture",
    "reconstruct_graph",
    "inspect_tensor",
    "summarize_weights",
    "capture_activations",
    "trace_generation",
    "build_circuit_map",
    "run_activation_ablation",
    "run_parameter_ablation",
    "analyze_weight_directions",
    "generate_patch",
    "evaluate_candidate",
    "optimize_patch",
    "validate_patch",
    "apply_patch",
    "rollback_patch",
    # WSP-X Intelligence & Surgery
    "profile_matrix_subspace",
    "profile_model_subspaces",
    "compute_spectral_regularization_delta",
    "project_delta_onto_null_space",
    "ParetoSurgeryPlanner",
    # Evaluation & Infrastructure
    "compare_models",
    "evaluate_metrics",
    "ModelLoader",
    "MCPServer",
    "ActivationTracer",
    # Domain Types
    "ModelFormat",
    "TensorRole",
    "AblationMode",
    "ModelMetadata",
    "TensorSummary",
    "GraphNode",
    "ComputationGraph",
    "ActivationTrace",
    "CircuitNode",
    "CircuitMap",
    "AblationResult",
    "PatchOperation",
    "PatchManifest",
    "EvaluationResult",
    "SubspaceProfile",
    "CapabilitySubspaceMap",
    "SurgeryAction",
    "SurgeryPlan",
]
