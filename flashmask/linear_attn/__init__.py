# -*- coding: utf-8 -*-
# flash-linear-attention Paddle migration entry point

try:
    import paddle
    from linear_attn.triton_utils import _is_package_installed

    # No torch environment: enable triton scope compat globally (zero runtime overhead)
    if not _is_package_installed("torch") and hasattr(paddle, "enable_compat"):
        paddle.enable_compat(scope={"triton"})
except Exception:
    pass

from linear_attn.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    chunk_gdn,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gdn,
)
from linear_attn.ops.kda import (
    chunk_kda,
    fused_recurrent_kda,
)

__all__ = [
    'chunk_gated_delta_rule',
    'chunk_gdn',
    'fused_recurrent_gated_delta_rule',
    'fused_recurrent_gdn',
    'chunk_kda',
    'fused_recurrent_kda',
]
