import csv
import json

import pytest

from ts_symbollm.results import FIELD_NAMES, ResultRecord, ResultsWriter


def _make_record(i: int) -> ResultRecord:
    return ResultRecord.create(
        model="qwen2.5:7b",
        representation="raw",
        dataset_id=f"dataset_{i}",
        params={"temperature": 0.1},
        prompt=f"prompt {i}",
        response=f"response {i}",
        latency_seconds=0.5 + i,
    )


def test_result_record_create_resolves_model_size_and_timestamp():
    record = _make_record(0)
    assert record.model_size == "small"  # qwen2.5:7b is in the shipped "small" tier
    assert record.timestamp  # non-empty, auto-filled


def test_result_record_create_honors_explicit_overrides():
    record = ResultRecord.create(
        model="some-custom-model",
        representation="symbolic",
        dataset_id="d",
        params=None,
        prompt="p",
        response="r",
        latency_seconds=1.0,
        model_size="custom-size",
        timestamp="2024-01-01T00:00:00",
    )
    assert record.model_size == "custom-size"
    assert record.timestamp == "2024-01-01T00:00:00"


def test_jsonl_write_produces_n_loadable_rows(tmp_path):
    path = tmp_path / "results.jsonl"
    writer = ResultsWriter(str(path))
    assert writer.format == "jsonl"

    for i in range(5):
        writer.write(_make_record(i))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    rows = [json.loads(line) for line in lines]
    assert [row["dataset_id"] for row in rows] == [f"dataset_{i}" for i in range(5)]
    assert rows[0]["params"] == {"temperature": 0.1}


def test_csv_write_produces_n_loadable_rows_with_one_header(tmp_path):
    path = tmp_path / "results.csv"
    writer = ResultsWriter(str(path))
    assert writer.format == "csv"

    for i in range(5):
        writer.write(_make_record(i))

    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        assert reader.fieldnames == FIELD_NAMES

    assert len(rows) == 5
    assert [row["dataset_id"] for row in rows] == [f"dataset_{i}" for i in range(5)]
    # params round-trips as a JSON string in CSV cells
    assert json.loads(rows[0]["params"]) == {"temperature": 0.1}


def test_format_can_be_forced_regardless_of_extension(tmp_path):
    path = tmp_path / "results.txt"
    writer = ResultsWriter(str(path), format="csv")
    assert writer.format == "csv"


def test_invalid_format_raises(tmp_path):
    with pytest.raises(ValueError):
        ResultsWriter(str(tmp_path / "results.jsonl"), format="xml")


def test_writer_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "results.jsonl"
    writer = ResultsWriter(str(path))
    writer.write(_make_record(0))
    assert path.exists()
