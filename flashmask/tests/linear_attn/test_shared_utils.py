import paddle


def test_input_guard_accepts_boolean_no_guard_contiguous():
    from linear_attn.utils import input_guard

    @input_guard(no_guard_contiguous=True)
    def identity(x):
        return x

    x = paddle.arange(8, dtype=paddle.float32).reshape([2, 4]).transpose([1, 0])
    y = identity(x)

    assert list(y.shape) == [4, 2]


def test_modules_package_exports_l2norm_symbols():
    import linear_attn.modules as modules

    assert hasattr(modules, 'L2Norm')
    assert hasattr(modules, 'l2norm')
    assert hasattr(modules, 'l2norm_fwd')
    assert hasattr(modules, 'l2norm_bwd')
    assert hasattr(modules, 'l2_norm')
    assert hasattr(modules, 'L2NormFunction')


def test_ops_utils_exports_core_helpers():
    import linear_attn.ops.utils as ops_utils

    assert hasattr(ops_utils, 'chunk_local_cumsum')
    assert hasattr(ops_utils, 'prepare_chunk_indices')
    assert hasattr(ops_utils, 'softplus')
    assert hasattr(ops_utils, 'solve_tril')
