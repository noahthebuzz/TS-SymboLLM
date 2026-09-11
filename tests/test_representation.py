import numpy as np

from ts_symbollm import representation as repr_mod


def _timestamps(n):
    return [f"t{i}" for i in range(n)]


def test_to_raw_returns_values_unchanged():
    data = [1.234, 5.678, 9.0]
    ts = _timestamps(3)
    assert repr_mod.to_raw(data, ts) == dict(zip(ts, data))


def test_to_rounded_honors_explicit_decimal_places():
    data = [1.234, 5.6789, 9.005]
    ts = _timestamps(3)
    for places in (0, 1, 2):
        result = repr_mod.to_rounded(data, ts, decimal_places=places)
        expected = dict(zip(ts, np.around(data, places).tolist()))
        assert result == expected


def test_to_rounded_defaults_to_zero_decimal_places():
    data = [1.4, 2.6, 9.9]
    ts = _timestamps(3)
    result = repr_mod.to_rounded(data, ts)
    expected = dict(zip(ts, np.around(data, 0).tolist()))
    assert result == expected


def test_default_num_symbols_divides_evenly():
    assert repr_mod.default_num_symbols(100) == 10
    assert repr_mod.default_num_symbols(50, segment_length=5) == 10


def test_default_num_symbols_rounds_up_when_not_divisible():
    assert repr_mod.default_num_symbols(95) == 10  # ceil(9.5)
    assert repr_mod.default_num_symbols(101) == 11  # ceil(10.1)


def test_default_num_symbols_minimum_is_one():
    assert repr_mod.default_num_symbols(5) == 1
    assert repr_mod.default_num_symbols(1) == 1
    assert repr_mod.default_num_symbols(0) == 1


def test_to_symbolic_default_params_matches_n_over_10_segments():
    data = [float(x) for x in range(100)]
    ts = _timestamps(100)
    result = repr_mod.to_symbolic(data, ts)
    assert len(result) == 10


def test_to_symbolic_explicit_overrides_are_honored():
    data = [float(x) for x in range(50)]
    ts = _timestamps(50)
    result = repr_mod.to_symbolic(data, ts, num_symbols=5, levels=4)
    assert len(result) == 5
    for value in result.values():
        assert 0 <= value < 4


def test_apply_dispatches_to_the_matching_representation():
    data = [1.0, 2.0, 3.0]
    ts = _timestamps(3)
    assert repr_mod.apply("raw", data, ts) == repr_mod.to_raw(data, ts)
    assert repr_mod.apply(repr_mod.Representation.ROUNDED, data, ts) == repr_mod.to_rounded(data, ts)
