import json

import pytest

from ts_symbollm.cli import load_time_series


# The four formats below are copied verbatim from the README's
# "Supported data formats" section.

CSV_SINGLE_SERIES = """timestamp,value
2024-03-01T08:00:00,48.2
2024-03-01T08:05:00,48.9
"""

CSV_MULTI_SERIES = """timestamp,series,value
2024-03-01T08:00:00,cpu_utilization,32.5
2024-03-01T08:00:00,cpu_temperature,46.1
"""

JSON_SINGLE_SERIES = {
    "data": {
        "2024-03-01T08:00:00": 48.2,
        "2024-03-01T08:05:00": 48.9,
    }
}

JSON_MULTI_SERIES = {
    "series": [
        {
            "name": "cpu_utilization_percent",
            "data": {
                "2024-03-01T08:00:00": 32.5,
                "2024-03-01T08:05:00": 35.2,
            },
        },
        {
            "name": "cpu_temperature_c",
            "data": {
                "2024-03-01T08:00:00": 46.1,
                "2024-03-01T08:05:00": 46.8,
            },
        },
    ]
}


def test_load_time_series_csv_single_series(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text(CSV_SINGLE_SERIES, encoding="utf-8")

    series_data = load_time_series(str(path))

    assert list(series_data.keys()) == ["series_1"]
    assert series_data["series_1"] == [
        ("2024-03-01T08:00:00", 48.2),
        ("2024-03-01T08:05:00", 48.9),
    ]


def test_load_time_series_csv_multi_series(tmp_path):
    path = tmp_path / "multi.csv"
    path.write_text(CSV_MULTI_SERIES, encoding="utf-8")

    series_data = load_time_series(str(path))

    assert set(series_data.keys()) == {"cpu_utilization", "cpu_temperature"}
    assert series_data["cpu_utilization"] == [("2024-03-01T08:00:00", 32.5)]
    assert series_data["cpu_temperature"] == [("2024-03-01T08:00:00", 46.1)]


def test_load_time_series_csv_without_header(tmp_path):
    path = tmp_path / "headerless.csv"
    path.write_text("2024-03-01T08:00:00,1.5\n2024-03-01T08:05:00,2.5\n", encoding="utf-8")

    series_data = load_time_series(str(path))

    assert series_data == {
        "series_1": [
            ("2024-03-01T08:00:00", 1.5),
            ("2024-03-01T08:05:00", 2.5),
        ]
    }


def test_load_time_series_json_single_series(tmp_path):
    path = tmp_path / "single.json"
    path.write_text(json.dumps(JSON_SINGLE_SERIES), encoding="utf-8")

    series_data = load_time_series(str(path))

    assert list(series_data.keys()) == ["series_1"]
    assert series_data["series_1"] == [
        ("2024-03-01T08:00:00", 48.2),
        ("2024-03-01T08:05:00", 48.9),
    ]


def test_load_time_series_json_multi_series(tmp_path):
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(JSON_MULTI_SERIES), encoding="utf-8")

    series_data = load_time_series(str(path))

    assert set(series_data.keys()) == {"cpu_utilization_percent", "cpu_temperature_c"}
    assert series_data["cpu_utilization_percent"] == [
        ("2024-03-01T08:00:00", 32.5),
        ("2024-03-01T08:05:00", 35.2),
    ]
    assert series_data["cpu_temperature_c"] == [
        ("2024-03-01T08:00:00", 46.1),
        ("2024-03-01T08:05:00", 46.8),
    ]


def test_load_time_series_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_time_series(str(tmp_path / "does_not_exist.csv"))


def test_load_time_series_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("not a supported format", encoding="utf-8")
    with pytest.raises(ValueError):
        load_time_series(str(path))


def test_load_time_series_json_missing_data_key_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not_data": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_time_series(str(path))


def test_load_time_series_csv_non_numeric_value_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,value\n2024-03-01T08:00:00,not-a-number\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_time_series(str(path))
