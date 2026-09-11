from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe: this module only ever saves to file
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .config import config

DEFAULT_OUTPUT_DIR = "./plots/"

_COLORS = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown']


def get_output_dir(output_dir: str | None = None) -> str:
    '''
    Resolves the diagram output directory: an explicit argument wins, then
    the config file's "plotting.output_dir", then a hardcoded default.
    '''
    if output_dir is not None:
        return output_dir
    return config.get_plotting_config().get("output_dir", DEFAULT_OUTPUT_DIR)


def plot_series(
    series_data: Dict[str, List[Tuple[str, float]]],
    title: str,
    xlabel: str = "Time",
    ylabel: str = "Value",
    output_dir: str | None = None,
    filename: str | None = None,
) -> str:
    '''
    Plots one or more named series (the shape produced by
    helpers.load_time_series: {series_name: [(timestamp, value), ...]}) on
    a single figure and saves it as a PNG. Works the same way for a single
    series or several overlaid series with a legend, and scales correctly
    for value ranges of any magnitude.

    Returns the path the plot was saved to.
    '''
    output_dir = get_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 7))

    longest_timestamps: List[str] = []
    for i, (name, points) in enumerate(series_data.items()):
        if not points:
            continue
        timestamps = [point[0] for point in points]
        values = [point[1] for point in points]
        color = _COLORS[i % len(_COLORS)]
        ax.plot(timestamps, values, marker='o', color=color, label=name)
        if len(timestamps) > len(longest_timestamps):
            longest_timestamps = timestamps

    if len(longest_timestamps) >= 11:
        ax.set_xticks(longest_timestamps[::len(longest_timestamps) // 11])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which='both', linewidth='0.5', color='gray')
    # Let matplotlib pick sensible tick spacing for the actual value range,
    # instead of a fixed round(max*1.25, -1) scheme that only makes sense
    # for values roughly in the 0-100 range.
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    if len(series_data) > 1:
        ax.legend(loc='best', fontsize=10)
    plt.xticks(rotation=22.5)
    fig.tight_layout()

    # Guard against a caller passing a path-like title/filename (e.g. a raw
    # dataset path): os.path.join silently discards output_dir if the
    # second argument looks absolute, so strip path separators too, not
    # just spaces.
    safe_name = (filename or title).replace(" ", "_").replace("/", "_").replace("\\", "_")
    path = os.path.join(output_dir, f"{safe_name}_plot.png")
    fig.savefig(path)
    plt.close(fig)
    return path
