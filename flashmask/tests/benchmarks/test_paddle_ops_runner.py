from benchmarks.paddle_ops import registry
from benchmarks.paddle_ops.run import main


def test_registry_uses_linear_attn_imports():
    assert registry.get_op("chunk_kda").import_path == "linear_attn.ops.kda"
    assert registry.get_op("chunk_gdn").import_path == "linear_attn.ops.gated_delta_rule"


def test_list_cli_prints_registered_ops(capsys):
    main(["--list"])
    out = capsys.readouterr().out
    assert "chunk_kda" in out
    assert "recurrent_kda" in out
