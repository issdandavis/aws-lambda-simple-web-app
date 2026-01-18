"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               DEFINITION 4.3/4.4: HAL-ATTENTION                              ║
║══════════════════════════════════════════════════════════════════════════════║
║                                                                              ║
║  Harmonic Attention Layer: HAL-Attention(Q, K, V, d) = softmax(H_weight) · V ║
║                                                                              ║
║  Definition 4.3.1 - Harmonic Attention:                                      ║
║    H_weight(Q, K, d) = (QKᵀ / √d_k) ⊙ Λ(d)                                   ║
║                                                                              ║
║  Definition 4.4.1 - Coupling Matrix Λ:                                       ║
║    Λ(d_Q, d_K)[i,j] = R^(d_Q[i] · d_K[j])                                    ║
║                                                                              ║
║  Extends standard transformer attention with harmonic weighting derived      ║
║  from the H(d, R) scaling law for dimension-aware resonance.                 ║
║                                                                              ║
║  Document ID: AETHER-SPEC-2026-001                                           ║
║  Section: 4 (HAL-Attention Mathematics)                                      ║
║  Author: Isaac Davis                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Re-export from hal_attention module
from ..hal_attention import (
    # Configuration
    HALConfig,
    AttentionOutput,

    # Core Functions
    harmonic_coupling_matrix,
    assign_dimension_depths,
    hal_attention,
    multi_head_hal_attention,

    # Layer Class
    HALAttentionLayer,

    # Utilities
    visualize_coupling_matrix,
    get_hal_stats,

    # Types
    Vector, Matrix, Tensor3D, Tensor2D,
)

__all__ = [
    'HALConfig',
    'AttentionOutput',
    'harmonic_coupling_matrix',
    'assign_dimension_depths',
    'hal_attention',
    'multi_head_hal_attention',
    'HALAttentionLayer',
    'visualize_coupling_matrix',
    'get_hal_stats',
    'Vector', 'Matrix', 'Tensor3D', 'Tensor2D',
]

AXIOM_ID = "4.3/4.4"
AXIOM_TITLE = "HAL-Attention (Harmonic Attention Layer)"
AXIOM_FORMULA = "HAL(Q, K, V, d) = softmax((QKᵀ/√d_k) ⊙ Λ(d)) · V"
