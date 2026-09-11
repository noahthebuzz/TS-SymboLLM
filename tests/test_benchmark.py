import csv
import json
from unittest.mock import patch

import pytest

from ts_symbollm import cli


def _fake_ollama_generate(response_text="a response"):
    def _generate(model, prompt, stream, options=None):
        return iter([{"response": response_text}])
    return _generate


def test_parse_list_splits_and_strips():
    assert cli._parse_list("a, b ,c") == ["a", "b", "c"]
    assert cli._parse_list("") == []


def test_resolve_dataset_path_prefers_bundled_example():
    assert cli._resolve_dataset_path("temperature") == cli.EXAMPLES["temperature"]
    assert cli._resolve_dataset_path("/some/path.csv") == "/some/path.csv"


def test_resolve_params_preset():
    assert cli._resolve_params_preset("default") is None
    assert cli._resolve_params_preset("rational") is not None
    assert cli._resolve_params_preset("creative") is not None
    with pytest.raises(ValueError):
        cli._resolve_params_preset("not-a-real-preset")


def test_benchmark_runs_all_representations_and_writes_one_row_each(tmp_path):
    results_path = tmp_path / "results.jsonl"
    parser = cli.build_benchmark_parser()
    args = parser.parse_args([
        "--data", "temperature",
        "--model", "qwen2.5:7b",
        "--representation", "raw,rounded,symbolic",
        "--results", str(results_path),
        "--no-plot",
    ])

    with patch("ts_symbollm.backends.ollama.ollama.generate", side_effect=_fake_ollama_generate()):
        cli.run_benchmark(args)

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert {row["representation"] for row in rows} == {"raw", "rounded", "symbolic"}
    assert all(row["dataset_id"] == "temperature" for row in rows)
    assert all(row["response"] == "a response" for row in rows)


def test_benchmark_sweeps_the_full_matrix(tmp_path):
    results_path = tmp_path / "results.csv"
    parser = cli.build_benchmark_parser()
    args = parser.parse_args([
        "--data", "temperature,cpu",
        "--model", "modelA,modelB",
        "--representation", "raw,symbolic",
        "--params", "default,rational",
        "--results", str(results_path),
        "--no-plot",
    ])

    with patch("ts_symbollm.backends.ollama.ollama.generate", side_effect=_fake_ollama_generate()):
        cli.run_benchmark(args)

    with open(results_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 16  # 2 datasets x 2 models x 2 representations x 2 params
    combos = {(r["dataset_id"], r["model"], r["representation"], r["params"]) for r in rows}
    assert len(combos) == 16


def test_benchmark_no_plot_skips_diagrams(tmp_path, monkeypatch):
    results_path = tmp_path / "results.jsonl"
    plot_dir = tmp_path / "plots"
    parser = cli.build_benchmark_parser()
    args = parser.parse_args([
        "--data", "temperature",
        "--model", "qwen2.5:7b",
        "--results", str(results_path),
        "--plot-dir", str(plot_dir),
        "--no-plot",
    ])

    with patch("ts_symbollm.backends.ollama.ollama.generate", side_effect=_fake_ollama_generate()):
        cli.run_benchmark(args)

    assert not plot_dir.exists()


def test_benchmark_generates_one_diagram_per_representation(tmp_path):
    results_path = tmp_path / "results.jsonl"
    plot_dir = tmp_path / "plots"
    parser = cli.build_benchmark_parser()
    args = parser.parse_args([
        "--data", "temperature",
        "--model", "qwen2.5:7b",
        "--representation", "raw,rounded,symbolic",
        "--results", str(results_path),
        "--plot-dir", str(plot_dir),
    ])

    with patch("ts_symbollm.backends.ollama.ollama.generate", side_effect=_fake_ollama_generate()):
        cli.run_benchmark(args)

    saved = sorted(p.name for p in plot_dir.glob("*.png"))
    assert saved == [
        "temperature_(raw)_plot.png",
        "temperature_(rounded)_plot.png",
        "temperature_(symbolic)_plot.png",
    ]


def test_benchmark_score_flag_attaches_score_for_known_dataset(tmp_path):
    results_path = tmp_path / "results.jsonl"
    parser = cli.build_benchmark_parser()
    args = parser.parse_args([
        "--data", "temperature",
        "--model", "qwen2.5:7b",
        "--results", str(results_path),
        "--no-plot",
        "--score",
    ])

    with patch(
        "ts_symbollm.backends.ollama.ollama.generate",
        side_effect=_fake_ollama_generate("Temperature shows a steady rising trend."),
    ):
        cli.run_benchmark(args)

    row = json.loads(results_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["score"] == 1.0


def test_benchmark_score_flag_leaves_unscored_dataset_alone(tmp_path):
    results_path = tmp_path / "results.jsonl"
    parser = cli.build_benchmark_parser()
    args = parser.parse_args([
        "--data", "cpu",  # no EXAMPLE_METADATA entry -> nothing to score against
        "--model", "qwen2.5:7b",
        "--results", str(results_path),
        "--no-plot",
        "--score",
    ])

    with patch("ts_symbollm.backends.ollama.ollama.generate", side_effect=_fake_ollama_generate()):
        cli.run_benchmark(args)

    row = json.loads(results_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["score"] is None


def test_benchmark_requires_no_interactive_input(tmp_path, monkeypatch):
    '''A scripted run must never call input() -- simulate a hostile
    environment where input() would raise if invoked.'''
    def _explode(*args, **kwargs):
        raise AssertionError("input() should never be called during a benchmark run")

    monkeypatch.setattr("builtins.input", _explode)

    results_path = tmp_path / "results.jsonl"
    parser = cli.build_benchmark_parser()
    args = parser.parse_args([
        "--data", "temperature",
        "--model", "qwen2.5:7b",
        "--results", str(results_path),
        "--no-plot",
    ])
    with patch("ts_symbollm.backends.ollama.ollama.generate", side_effect=_fake_ollama_generate()):
        cli.run_benchmark(args)  # would raise via the monkeypatched input() if it were ever called


def test_benchmark_with_a_full_file_path_saves_plot_inside_plot_dir(tmp_path):
    '''
    Regression test: passing a full path (not a bundled example name) as
    --data must not leak path separators into the plot title/filename --
    os.path.join silently discards the output dir if the filename looks
    absolute, which previously caused plots to be written next to the
    source data file instead of into --plot-dir.
    '''
    results_path = tmp_path / "results.jsonl"
    plot_dir = tmp_path / "plots"
    data_path = cli.EXAMPLES["temperature"]

    parser = cli.build_benchmark_parser()
    args = parser.parse_args([
        "--data", data_path,
        "--model", "qwen2.5:7b",
        "--results", str(results_path),
        "--plot-dir", str(plot_dir),
    ])

    with patch("ts_symbollm.backends.ollama.ollama.generate", side_effect=_fake_ollama_generate()):
        cli.run_benchmark(args)

    saved = list(plot_dir.glob("*.png"))
    assert len(saved) == 1
    assert saved[0].parent == plot_dir
    assert "/" not in saved[0].name


def test_main_dispatches_benchmark_subcommand(tmp_path, monkeypatch):
    results_path = tmp_path / "results.jsonl"
    monkeypatch.setattr("sys.argv", [
        "ts-symbollm", "benchmark",
        "--data", "temperature",
        "--model", "qwen2.5:7b",
        "--results", str(results_path),
        "--no-plot",
    ])
    with patch("ts_symbollm.backends.ollama.ollama.generate", side_effect=_fake_ollama_generate()):
        cli.main()
    assert results_path.exists()
