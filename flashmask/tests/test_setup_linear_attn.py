import setup_linear_attn


def test_setup_linear_attn_only_packages_linear_attn():
    kwargs = setup_linear_attn.build_setup_kwargs()

    assert kwargs["name"] == "flashmask-linear-attn-dev"
    assert "linear_attn" in kwargs["packages"]
    assert all(pkg == "linear_attn" or pkg.startswith("linear_attn.") for pkg in kwargs["packages"])
    assert "benchmarks" not in kwargs["packages"]
    assert "tests" not in kwargs["packages"]
    assert "einops" in kwargs["install_requires"]
    assert "typing_extensions" in kwargs["install_requires"]


def test_setup_linear_attn_declares_test_extra():
    kwargs = setup_linear_attn.build_setup_kwargs()

    assert kwargs["extras_require"]["test"] == ["pytest"]
