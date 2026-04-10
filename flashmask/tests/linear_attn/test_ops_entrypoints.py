def test_gated_delta_rule_exports_core_entrypoints():
    from linear_attn.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        chunk_gdn,
        fused_recurrent_gated_delta_rule,
        fused_recurrent_gdn,
        naive_recurrent_gated_delta_rule,
    )

    assert callable(chunk_gated_delta_rule)
    assert chunk_gdn is chunk_gated_delta_rule
    assert callable(fused_recurrent_gated_delta_rule)
    assert fused_recurrent_gdn is fused_recurrent_gated_delta_rule
    assert callable(naive_recurrent_gated_delta_rule)


def test_kda_exports_core_entrypoints():
    from linear_attn.ops.kda import chunk_kda, fused_recurrent_kda

    assert callable(chunk_kda)
    assert callable(fused_recurrent_kda)


def test_linear_attn_top_level_exports_core_entrypoints():
    import linear_attn

    assert callable(linear_attn.chunk_gated_delta_rule)
    assert callable(linear_attn.chunk_gdn)
    assert callable(linear_attn.fused_recurrent_gated_delta_rule)
    assert callable(linear_attn.fused_recurrent_gdn)
    assert callable(linear_attn.chunk_kda)
    assert callable(linear_attn.fused_recurrent_kda)
