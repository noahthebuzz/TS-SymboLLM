import pytest

from ts_symbollm.prompt import PromptBuilder


def test_single_section_uses_unlabeled_data_headers():
    prompt = (
        PromptBuilder(task="Interpret the series.", desired_output="A short summary.")
        .add_data(context="CPU temperature in Celsius.", data="08:00: 46\n08:05: 47")
        .build()
    )
    assert prompt == (
        "Task:\nInterpret the series.\n"
        "Data context:\nCPU temperature in Celsius.\n"
        "Data:\n08:00: 46\n08:05: 47\n"
        "Desired output format:\nA short summary.\n"
    )


def test_multi_section_uses_numbered_data_headers():
    prompt = (
        PromptBuilder(task="Compare the series.", desired_output="Explain the relationship.")
        .add_data(context="Capacity (%)", data="08:00: 30")
        .add_data(context="Temperature (°C)", data="08:00: 46")
        .build()
    )
    assert "Data 1 context:\nCapacity (%)\n" in prompt
    assert "Data 1:\n08:00: 30\n" in prompt
    assert "Data 2 context:\nTemperature (°C)\n" in prompt
    assert "Data 2:\n08:00: 46\n" in prompt
    assert prompt.index("Data 1") < prompt.index("Data 2") < prompt.index("Desired output format:")


def test_build_without_any_data_section_raises():
    with pytest.raises(ValueError):
        PromptBuilder(task="t", desired_output="o").build()


def test_add_series_raw_renders_timestamp_value_lines():
    points = [("t0", 1.0), ("t1", 2.0)]
    prompt = (
        PromptBuilder(task="t", desired_output="o")
        .add_series(context="ctx", points=points, representation="raw")
        .build()
    )
    assert "t0: 1.0" in prompt
    assert "t1: 2.0" in prompt


def test_add_series_symbolic_renders_letter_string_not_numbers():
    points = [(f"t{i}", float(i)) for i in range(20)]
    prompt = (
        PromptBuilder(task="t", desired_output="o")
        .add_series(context="ctx", points=points, representation="symbolic", num_symbols=4, levels=4)
        .build()
    )
    lines = prompt.splitlines()
    letters = lines[lines.index("Data:") + 1]
    assert len(letters) == 4
    assert letters.isalpha()
    assert all(0 <= ord(letter) - ord('a') < 4 for letter in letters)


def test_add_series_rounded_respects_decimal_places():
    points = [("t0", 1.234), ("t1", 5.678)]
    prompt = (
        PromptBuilder(task="t", desired_output="o")
        .add_series(context="ctx", points=points, representation="rounded", decimal_places=1)
        .build()
    )
    assert "t0: 1.2" in prompt
    assert "t1: 5.7" in prompt


def test_multiple_add_series_calls_produce_a_multi_section_prompt():
    cap_points = [("t0", 30.0), ("t1", 40.0)]
    temp_points = [("t0", 46.0), ("t1", 47.0)]
    prompt = (
        PromptBuilder(task="Compare capacity and temperature.", desired_output="Explain the relationship.")
        .add_series(context="Capacity (%)", points=cap_points, representation="raw")
        .add_series(context="Temperature (°C)", points=temp_points, representation="raw")
        .build()
    )
    assert "Data 1 context:\nCapacity (%)\n" in prompt
    assert "Data 2 context:\nTemperature (°C)\n" in prompt
    assert "t0: 30.0" in prompt
    assert "t0: 46.0" in prompt
