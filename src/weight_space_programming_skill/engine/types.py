"""Typed data structures and domain models for the Weight-Space Programming (WSP) engine."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class ModelFormat(str, enum.Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    UNKNOWN = "unknown"


class TensorRole(str, enum.Enum):
    EMBEDDING = "embedding"
    ATTENTION_QUERY = "attention_query"
    ATTENTION_KEY = "attention_key"
    ATTENTION_VALUE = "attention_value"
    ATTENTION_OUTPUT = "attention_output"
    ATTENTION_QKV_FUSED = "attention_qkv_fused"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    NORMALIZATION = "normalization"
    OUTPUT_HEAD = "output_head"
    ROUTING_MOE = "routing_moe"
    CONVOLUTION = "convolution"
    RECURRENT = "recurrent"
    GENERIC = "generic"


class AblationMode(str, enum.Enum):
    ZERO = "zero"
    MEAN = "mean"
    NOISE = "noise"
    FREEZE = "freeze"


@dataclass
class ModelMetadata:
    path: str
    format: ModelFormat
    file_size_bytes: int
    n_tensors: int
    total_parameters: int
    architecture_guess: str
    layer_count: int
    data_types: Dict[str, int] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorSummary:
    name: str
    shape: List[int]
    dtype: str
    numel: int
    role: TensorRole
    component_family: str
    layer_index: Optional[int]
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    sparsity: float = 0.0
    l2_norm: float = 0.0
    finite_fraction: float = 1.0
    quantiles: List[float] = field(default_factory=list)


@dataclass
class GraphNode:
    name: str
    module_type: str
    layer_index: Optional[int]
    input_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    shapes: Dict[str, List[int]] = field(default_factory=dict)
    is_residual: bool = False


@dataclass
class ComputationGraph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    shared_parameters: List[Tuple[str, str]] = field(default_factory=list)
    total_layers: int = 0
    input_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    output_dim: Optional[int] = None


@dataclass
class ActivationTrace:
    module_name: str
    step_index: int
    mean: float
    std: float
    max_abs: float
    l2_norm: float
    sparsity: float
    subspace_variance: Optional[float] = None


@dataclass
class CircuitNode:
    component: str
    layer_index: int
    importance_score: float
    ablation_impact: float
    hypothesized_role: str


@dataclass
class CircuitMap:
    target_behavior: str
    critical_nodes: List[CircuitNode] = field(default_factory=list)
    causal_edges: List[Tuple[str, str, float]] = field(default_factory=list)
    summary: str = ""


@dataclass
class AblationResult:
    target_component: str
    mode: AblationMode
    baseline_score: float
    ablated_score: float
    relative_change: float
    is_statistically_significant: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchOperation:
    tensor_name: str
    operation: str  # e.g., "add", "replace", "low_rank_update"
    shape: List[int]
    data: Optional[Any] = None
    alpha: float = 1.0
    rank: Optional[int] = None


@dataclass
class PatchManifest:
    patch_id: str
    target_model_path: str
    base_model_hash: str
    created_at: str
    author: str
    description: str
    operations: List[PatchOperation] = field(default_factory=list)
    verification_hash: Optional[str] = None
    is_reversible: bool = True


@dataclass
class EvaluationResult:
    baseline_metric: float
    candidate_metric: float
    delta: float
    p_value: Optional[float]
    passed_regression_gate: bool
    regression_summary: Dict[str, float] = field(default_factory=dict)
