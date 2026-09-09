"""EthicalGuard —— 跨临床场景的伦理决策平衡与韧性评估框架。"""

__version__ = "0.1.0"

from .types import (
    PRINCIPLES,
    HALF_WEIGHTS,
    PrincipleVector,
    Proposal,
    Scenario,
    Constraint,
    ConstraintKind,
    FourBoxState,
    NegotiationResult,
    ResilienceReport,
    Perturbation,
)

__all__ = [
    "__version__",
    "PRINCIPLES",
    "HALF_WEIGHTS",
    "PrincipleVector",
    "Proposal",
    "Scenario",
    "Constraint",
    "ConstraintKind",
    "FourBoxState",
    "NegotiationResult",
    "ResilienceReport",
    "Perturbation",
]
