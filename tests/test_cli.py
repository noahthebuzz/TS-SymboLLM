from ts_symbollm import cli


def _series(n=10):
    return {"series_1": [(f"t{i}", float(i)) for i in range(n)]}


def test_apply_representation_to_series_raw_is_unchanged():
    series_data = _series()
    result = cli.apply_representation_to_series(series_data, representation="raw")
    assert result == series_data


def test_apply_representation_to_series_rounded_rounds_values():
    series_data = {"series_1": [("t0", 1.6), ("t1", 2.4)]}
    result = cli.apply_representation_to_series(series_data, representation="rounded", decimal_places=0)
    assert result["series_1"] == [("t0", 2.0), ("t1", 2.0)]


def test_apply_representation_to_series_symbolic_compresses_length():
    series_data = _series(50)
    result = cli.apply_representation_to_series(series_data, representation="symbolic", num_symbols=5)
    assert len(result["series_1"]) == 5


def test_apply_representation_to_series_handles_empty_series():
    result = cli.apply_representation_to_series({"empty": []}, representation="raw")
    assert result == {"empty": []}


def test_format_series_data_symbolic_renders_single_letter_string_per_series():
    series_data = {"series_1": [("t0", 0), ("t1", 1), ("t2", 2)]}
    text = cli.format_series_data(series_data, max_points=None, representation="symbolic")
    assert text == "series_1: abc"


def test_format_series_data_raw_lists_one_line_per_point():
    series_data = {"series_1": [("t0", 1.5), ("t1", 2.5)]}
    text = cli.format_series_data(series_data, max_points=None, representation="raw")
    assert "series_1:" in text
    assert "  t0: 1.5" in text
    assert "  t1: 2.5" in text
