from __future__ import annotations

import math
from enum import Enum

import numpy as np
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from .config import config


class Representation(str, Enum):
    RAW = "raw"
    ROUNDED = "rounded"
    SYMBOLIC = "symbolic"


DEFAULT_DECIMAL_PLACES = 0
DEFAULT_LEVELS = 10
DEFAULT_SEGMENT_LENGTH = 10


def _defaults() -> dict:
    return config.get_representation_defaults()


def _decimal_places(decimal_places: int | None) -> int:
    if decimal_places is not None:
        return decimal_places
    return _defaults().get("rounded_decimal_places", DEFAULT_DECIMAL_PLACES)


def _levels(levels: int | None) -> int:
    if levels is not None:
        return levels
    return _defaults().get("symbolic_levels", DEFAULT_LEVELS)


def _segment_length(segment_length: int | None) -> int:
    if segment_length is not None:
        return segment_length
    return _defaults().get("symbolic_segment_length", DEFAULT_SEGMENT_LENGTH)


def resolve_decimal_places(decimal_places: int | None = None) -> int:
    '''Effective rounded-representation decimal places, resolving the config default if unset.'''
    return _decimal_places(decimal_places)


def resolve_levels(levels: int | None = None) -> int:
    '''Effective symbolic alphabet size, resolving the config default if unset.'''
    return _levels(levels)


def resolve_num_symbols(n: int, num_symbols: int | None = None, segment_length: int | None = None) -> int:
    '''Effective PAA segment count for a series of length n, resolving the ceil(n/segment_length) default if unset.'''
    if num_symbols is not None:
        return num_symbols
    return default_num_symbols(n, segment_length)


def default_num_symbols(n: int, segment_length: int | None = None) -> int:
    '''
    Default number of PAA segments for a series of length n: one symbol per
    `segment_length` raw points (rounded up), at least 1.
    '''
    segment_length = _segment_length(segment_length)
    return max(1, math.ceil(n / segment_length))


####################################################################
### RAW / ROUNDED / SYMBOLIC
####################################################################

def to_raw(data: list[float], timestamps: list[str]) -> dict[str, float]:
    return dict(zip(timestamps, data))


def to_rounded(data: list[float], timestamps: list[str], decimal_places: int | None = None) -> dict[str, float]:
    rounded = np.around(data, _decimal_places(decimal_places)).tolist()
    return dict(zip(timestamps, rounded))


def to_symbolic(
    data: list[float],
    timestamps: list[str],
    num_symbols: int | None = None,
    levels: int | None = None,
    segment_length: int | None = None,
) -> dict[str, float]:
    return approximate_tsd(data=data, timestamps=timestamps, num_symbols=num_symbols, levels=levels, segment_length=segment_length)


def apply(representation: Representation | str, data: list[float], timestamps: list[str], **kwargs) -> dict:
    '''
    Dispatches to the function for the given representation.

    Accepted kwargs: `decimal_places` (rounded), `num_symbols`/`levels`/
    `segment_length` (symbolic).
    '''
    representation = Representation(representation)
    if representation is Representation.RAW:
        return to_raw(data, timestamps)
    if representation is Representation.ROUNDED:
        return to_rounded(data, timestamps, decimal_places=kwargs.get("decimal_places"))
    if representation is Representation.SYMBOLIC:
        return to_symbolic(
            data,
            timestamps,
            num_symbols=kwargs.get("num_symbols"),
            levels=kwargs.get("levels"),
            segment_length=kwargs.get("segment_length"),
        )
    raise ValueError(f"Unknown representation: {representation}")


####################################################################
### PAA / SAX
####################################################################

def approximate_tsd(
    data: list[float],
    timestamps: list[str],
    num_symbols: int | None = None,
    levels: int | None = None,
    segment_length: int | None = None,
) -> dict[str, float]:
    '''
    Compresses a series via Piecewise Aggregate Approximation (PAA) and
    Symbolic Aggregate approXimation (SAX).

    `num_symbols` is the number of PAA segments the series is compressed
    to; if not given, it defaults to `ceil(len(data) / segment_length)`.
    `levels` is the SAX alphabet size / quantization depth.
    '''
    if num_symbols is None:
        num_symbols = default_num_symbols(len(data), segment_length)
    levels = _levels(levels)

    dataset = np.array(data).reshape(1, -1)

    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
    normalized_dataset = scaler.fit_transform(dataset)
    sax = SymbolicAggregateApproximation(n_segments=num_symbols, alphabet_size_avg=levels)
    sax_values = sax.fit_transform(normalized_dataset)

    reduced_timestamps = timestamps[::round(len(timestamps) / num_symbols)]

    return dict(zip(reduced_timestamps, sax_values[0].ravel().tolist()))


def get_sax_string(data: dict) -> dict[str, str]:
    keys = []
    values = []
    for datum in data.values():
        for key in datum.keys():
            keys.append(key)
        for value in datum.values():
            values.append(value)

    alphabet = 'abcdefghijklmnopqrstuvwxyz'[:len(values)]
    sax_string = [''.join([alphabet[int(i)] for i in values])][0]

    return dict(zip(keys, sax_string))


def get_sax_values(data: dict) -> dict[str, int]:
    keys = []
    values = []
    for datum in data.values():
        for key in datum.keys():
            keys.append(key)
        for value in datum.values():
            values.append(value)

    # match the letter in values to the corresponding number
    # like a = 1, b = 2, c = 3, ...
    sax_values = []
    for i in values:
        sax_values.append(ord(i) - ord('a') + 1)

    return dict(zip(keys, sax_values))
