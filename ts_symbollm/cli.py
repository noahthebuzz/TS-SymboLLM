from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

from . import plotting
from .backends.ollama import OllamaBackend
from .config import config
from .prompt import PromptBuilder
from .representation import Representation, apply as apply_representation, resolve_decimal_places
from .results import ResultRecord, ResultsWriter
from .scoring import TrendKeywordScorer

# examples/ ships at the repo root, not inside the package, so this only
# resolves for an editable install (`pip install -e .`) run from a checkout.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = {
    "temperature": os.path.join(REPO_ROOT, "examples", "data", "temperature.csv"),
    "cpu": os.path.join(REPO_ROOT, "examples", "data", "cpu_multi.json"),
}

# Ground-truth metadata for bundled examples with a known, scoreable trend
# (see ts_symbollm.scoring.KNOWN_TREND_PATTERNS). Datasets not listed here
# simply aren't scored, even with --score, since there's no known ground
# truth to check a response against.
EXAMPLE_METADATA = {
    "temperature": {"trend": "increasing"},
}

DEFAULT_RESULTS_PATH = os.path.join(".", "results", "results.jsonl")

DEFAULT_BENCHMARK_DESIRED_OUTPUT = (
    "Key patterns and trends; notable anomalies or shifts; likely explanations if possible; "
    "and suggested next checks or follow-up analysis. If the data is insufficient, say so and "
    "explain what is missing."
)

DEFAULT_PROMPT_TEMPLATE = """You are a data analyst helping interpret time series data.

Question:
{question}

Series summary:
{summary}

Sampled data points:
{data}

Please respond with:
- Key patterns and trends
- Notable anomalies or shifts
- Likely explanations (if possible)
- Suggested next checks or follow-up analysis
If the data is insufficient, say so and explain what is missing.
"""


def _find_key(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {key.lower(): key for key in keys}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _coerce_float(value: object, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Value for '{field}' is not numeric: {value!r}") from None


def _format_value(value: float | int, decimal_places: int | None = None) -> str:
    numeric = float(value)
    if decimal_places is not None:
        formatted = f"{numeric:.{max(decimal_places, 0)}f}"
        return formatted.rstrip("0").rstrip(".") if decimal_places > 0 else formatted
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _symbol_to_letter(value: float) -> str:
    return chr(ord('a') + int(round(value)))


def apply_representation_to_series(
    series_data: Dict[str, List[Tuple[str, float]]],
    representation: str,
    decimal_places: int | None = None,
    num_symbols: int | None = None,
    levels: int | None = None,
) -> Dict[str, List[Tuple[str, float]]]:
    result: Dict[str, List[Tuple[str, float]]] = {}
    for name, points in series_data.items():
        if not points:
            result[name] = []
            continue
        timestamps = [point[0] for point in points]
        values = [point[1] for point in points]
        transformed = apply_representation(
            representation,
            values,
            timestamps,
            decimal_places=decimal_places,
            num_symbols=num_symbols,
            levels=levels,
        )
        result[name] = list(transformed.items())
    return result


def _downsample(points: List[Tuple[str, float]], max_points: int | None) -> List[Tuple[str, float]]:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled[-1] = points[-1]
    return sampled


def _load_csv(path: str) -> Dict[str, List[Tuple[str, float]]]:
    with open(path, newline="", encoding="utf-8") as file:
        sample = file.read(2048)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        has_header = csv.Sniffer().has_header(sample)

        if has_header:
            reader = csv.DictReader(file, dialect=dialect)
            if not reader.fieldnames:
                raise ValueError("CSV header row is missing field names.")
            time_key = _find_key(reader.fieldnames, ["timestamp", "time", "date", "datetime"])
            value_key = _find_key(reader.fieldnames, ["value", "reading", "measurement", "metric", "val"])
            series_key = _find_key(reader.fieldnames, ["series", "name", "metric"])
            if time_key is None:
                time_key = reader.fieldnames[0]
            if value_key is None:
                if len(reader.fieldnames) < 2:
                    raise ValueError("CSV must include at least two columns for time and value.")
                value_key = reader.fieldnames[1]
            series_data: Dict[str, List[Tuple[str, float]]] = {}
            for row in reader:
                if not row:
                    continue
                series_name = row.get(series_key) if series_key else "series_1"
                timestamp = row.get(time_key, "").strip()
                value = _coerce_float(row.get(value_key), value_key)
                series_data.setdefault(series_name or "series_1", []).append((timestamp, value))
            return series_data

        reader = csv.reader(file, dialect=dialect)
        series_data = {"series_1": []}
        for row in reader:
            if not row or len(row) < 2:
                continue
            series_data["series_1"].append((row[0].strip(), _coerce_float(row[1], "value")))
        return series_data


def _series_from_mapping(name: str, mapping: Dict[object, object]) -> List[Tuple[str, float]]:
    points = []
    for timestamp, value in mapping.items():
        points.append((str(timestamp), _coerce_float(value, name)))
    return points


def _load_json(path: str) -> Dict[str, List[Tuple[str, float]]]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "series" in payload:
        series_payload = payload["series"]
        if isinstance(series_payload, dict):
            return {name: _series_from_mapping(name, data) for name, data in series_payload.items()}
        if isinstance(series_payload, list):
            series_data: Dict[str, List[Tuple[str, float]]] = {}
            for index, entry in enumerate(series_payload, start=1):
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or f"series_{index}"
                data = entry.get("data")
                if isinstance(data, dict):
                    series_data[name] = _series_from_mapping(name, data)
            if series_data:
                return series_data

    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        return {"series_1": _series_from_mapping("series_1", data)}

    if isinstance(data, list):
        series_data = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            time_key = _find_key(entry.keys(), ["timestamp", "time", "date", "datetime"])
            value_key = _find_key(entry.keys(), ["value", "reading", "measurement", "metric", "val"])
            series_key = _find_key(entry.keys(), ["series", "name", "metric"])
            if time_key is None or value_key is None:
                raise ValueError("JSON list entries must include timestamp and value fields.")
            series_name = entry.get(series_key) if series_key else "series_1"
            timestamp = str(entry.get(time_key))
            value = _coerce_float(entry.get(value_key), value_key)
            series_data.setdefault(series_name or "series_1", []).append((timestamp, value))
        if series_data:
            return series_data

    raise ValueError("Unsupported JSON format. Provide a 'series' object or data list.")


def load_time_series(path: str) -> Dict[str, List[Tuple[str, float]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.lower().endswith(".csv"):
        return _load_csv(path)
    if path.lower().endswith(".json"):
        return _load_json(path)
    raise ValueError("Unsupported file type. Use .csv or .json.")


def summarize_series(series_data: Dict[str, List[Tuple[str, float]]]) -> str:
    lines = []
    for name, points in series_data.items():
        if not points:
            continue
        values = [value for _, value in points]
        start_ts, end_ts = points[0][0], points[-1][0]
        avg_value = sum(values) / len(values)
        lines.append(
            f"- {name}: {len(points)} points from {start_ts} to {end_ts}; "
            f"min={_format_value(min(values))}, max={_format_value(max(values))}, "
            f"avg={_format_value(avg_value)}"
        )
    return "\n".join(lines)


def format_series_data(
    series_data: Dict[str, List[Tuple[str, float]]],
    max_points: int | None,
    representation: str = Representation.RAW.value,
    decimal_places: int | None = None,
) -> str:
    lines = []
    for name, points in series_data.items():
        if representation == Representation.SYMBOLIC.value:
            # Already compressed to num_symbols by apply_representation_to_series,
            # so render as a single compact SAX letter string, not one line per point.
            letters = "".join(_symbol_to_letter(value) for _, value in points)
            lines.append(f"{name}: {letters}")
            continue

        lines.append(f"{name}:")
        point_decimal_places = decimal_places if representation == Representation.ROUNDED.value else None
        for timestamp, value in _downsample(points, max_points):
            lines.append(f"  {timestamp}: {_format_value(value, point_decimal_places)}")
    return "\n".join(lines)


def load_prompt_template(path: str | None) -> str:
    if not path:
        return DEFAULT_PROMPT_TEMPLATE
    with open(path, encoding="utf-8") as file:
        return file.read()


def list_examples() -> None:
    print("Available examples:")
    for name, path in EXAMPLES.items():
        print(f"  - {name}: {os.path.relpath(path, REPO_ROOT)}")


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_dataset_path(token: str) -> str:
    return EXAMPLES.get(token, token)


def _dataset_label(token: str) -> str:
    '''A short, filesystem-safe label for a dataset token, for use in plot
    titles/filenames -- a bundled example name as-is, or just the
    filename (no directory, no extension) for an arbitrary path.'''
    if token in EXAMPLES:
        return token
    return os.path.splitext(os.path.basename(token))[0]


def _resolve_params_preset(name: str) -> Optional[dict]:
    if name == "default":
        return None
    if name == "rational":
        return config.get_ollama_parameter(isRational=True)
    if name == "creative":
        return config.get_ollama_parameter(isRational=False)
    raise ValueError(f"Unknown --params preset: {name!r} (use default, rational, or creative)")


def _print_summary_table(records: List[ResultRecord]) -> None:
    headers = ["dataset", "representation", "model", "latency_s", "score"]
    rows = [
        [
            record.dataset_id,
            record.representation,
            record.model,
            f"{record.latency_seconds:.2f}",
            "-" if record.score is None else f"{record.score:.2f}",
        ]
        for record in records
    ]
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
        for i, header in enumerate(headers)
    ]

    def _format_row(cells: List[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(_format_row(headers))
    print(_format_row(["-" * width for width in widths]))
    for row in rows:
        print(_format_row(row))


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ts-symbollm benchmark",
        description="Run a model x representation x dataset benchmark matrix, writing one structured result row per run.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Comma-separated dataset paths and/or bundled example names (see --list-examples).",
    )
    parser.add_argument("--model", required=True, help="Comma-separated model names to test.")
    parser.add_argument(
        "--representation",
        default=Representation.RAW.value,
        help="Comma-separated representations to sweep: raw,rounded,symbolic (default: raw).",
    )
    parser.add_argument(
        "--params",
        default="default",
        help="Comma-separated sampling-parameter presets to sweep: default,rational,creative (default: default).",
    )
    parser.add_argument(
        "--question",
        default="Interpret the series and explain patterns, anomalies, and potential causes.",
        help="Task/question included in the prompt.",
    )
    parser.add_argument(
        "--paa-segments",
        type=int,
        help="PAA segments for symbolic representation (default: ceil(n/10), overridable via config).",
    )
    parser.add_argument(
        "--alphabet-size",
        type=int,
        help="SAX alphabet size for symbolic representation (default: from config).",
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to the structured results file, JSONL or CSV by extension (default: {DEFAULT_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--plot-dir",
        help=f"Directory to save diagrams to (default: {plotting.DEFAULT_OUTPUT_DIR}, overridable via config).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating diagrams for this run.")
    parser.add_argument(
        "--score",
        action="store_true",
        help="Enable automatic scoring for datasets with known ground truth (see EXAMPLE_METADATA).",
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> None:
    dataset_tokens = _parse_list(args.data)
    models = _parse_list(args.model)
    representations = _parse_list(args.representation)
    params_presets = _parse_list(args.params)

    backend = OllamaBackend()
    writer = ResultsWriter(args.results)
    scorer = TrendKeywordScorer() if args.score else None

    records: List[ResultRecord] = []

    for dataset_token in dataset_tokens:
        dataset_path = _resolve_dataset_path(dataset_token)
        series_data = load_time_series(dataset_path)
        dataset_metadata = EXAMPLE_METADATA.get(dataset_token, {})

        for representation in representations:
            if not args.no_plot:
                represented_series_data = apply_representation_to_series(
                    series_data,
                    representation=representation,
                    num_symbols=args.paa_segments,
                    levels=args.alphabet_size,
                )
                plot_path = plotting.plot_series(
                    represented_series_data,
                    title=f"{_dataset_label(dataset_token)} ({representation})",
                    output_dir=args.plot_dir,
                )
                print(f"[PLOT] Saved diagram to {plot_path}")

            builder = PromptBuilder(task=args.question, desired_output=DEFAULT_BENCHMARK_DESIRED_OUTPUT)
            for name, points in series_data.items():
                builder.add_series(
                    context=name,
                    points=points,
                    representation=representation,
                    num_symbols=args.paa_segments,
                    levels=args.alphabet_size,
                )
            prompt = builder.build()

            for model in models:
                for params_name in params_presets:
                    params = _resolve_params_preset(params_name)
                    print(f"\n[BENCHMARK] dataset={dataset_token} representation={representation} model={model} params={params_name}\n")
                    start_time = time.time()
                    response = backend.generate(model=model, prompt=prompt, params=params)
                    latency_seconds = time.time() - start_time

                    score_result = None
                    if scorer is not None and dataset_metadata.get("trend"):
                        score_result = scorer.score(dataset_metadata, prompt, response)

                    record = ResultRecord.create(
                        model=model,
                        representation=representation,
                        dataset_id=dataset_token,
                        params=params,
                        prompt=prompt,
                        response=response,
                        latency_seconds=latency_seconds,
                        score=score_result.score if score_result else None,
                        score_notes=score_result.notes if score_result else None,
                    )
                    writer.write(record)
                    records.append(record)

    print(f"\n[BENCHMARK] wrote {len(records)} result(s) to {args.results}\n")
    _print_summary_table(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze time series data with a local Ollama model. "
            "Run `ts-symbollm benchmark --help` for the model x representation x dataset benchmark runner."
        ),
    )
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name to use.")
    parser.add_argument("--data", help="Path to a CSV or JSON time series file.")
    parser.add_argument("--example", choices=sorted(EXAMPLES.keys()), help="Run a bundled example dataset.")
    parser.add_argument("--prompt", help="Path to a custom prompt template file.")
    parser.add_argument(
        "--question",
        default="Interpret the series and explain patterns, anomalies, and potential causes.",
        help="Question or task to include in the prompt.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120,
        help="Maximum number of points per series to include in the prompt.",
    )
    parser.add_argument("--output", help="Write the model response to a file.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt before sending.")
    parser.add_argument("--list-examples", action="store_true", help="List available example datasets.")
    parser.add_argument(
        "--representation",
        choices=[r.value for r in Representation],
        default=Representation.RAW.value,
        help="How to represent series values in the prompt (default: raw).",
    )
    parser.add_argument(
        "--paa-segments",
        type=int,
        help="Number of PAA segments for --representation symbolic (default: ceil(n/10), overridable via config).",
    )
    parser.add_argument(
        "--alphabet-size",
        type=int,
        help="SAX alphabet size for --representation symbolic (default: from config).",
    )
    parser.add_argument(
        "--plot-dir",
        help=f"Directory to save the generated diagram to (default: {plotting.DEFAULT_OUTPUT_DIR}, overridable via config).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating a diagram for this run.")
    return parser


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "benchmark":
        run_benchmark(build_benchmark_parser().parse_args(argv[1:]))
        return

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_examples:
        list_examples()
        return

    if bool(args.data) == bool(args.example):
        parser.error("Provide either --data or --example (but not both).")

    data_path = EXAMPLES.get(args.example) if args.example else args.data
    if not data_path:
        parser.error("No data path resolved. Use --data or --example.")

    series_data = load_time_series(data_path)
    represented_series_data = apply_representation_to_series(
        series_data,
        representation=args.representation,
        num_symbols=args.paa_segments,
        levels=args.alphabet_size,
    )

    if not args.no_plot:
        dataset_label = args.example or os.path.splitext(os.path.basename(data_path))[0]
        plot_path = plotting.plot_series(
            represented_series_data,
            title=f"{dataset_label} ({args.representation})",
            output_dir=args.plot_dir,
        )
        print(f"[PLOT] Saved diagram to {plot_path}")

    # The summary always describes the real underlying values, even in
    # symbolic mode, so the model has the true scale alongside the compact
    # SAX string (which alone carries no absolute-value information).
    summary = summarize_series(series_data)
    data_text = format_series_data(
        represented_series_data,
        args.max_points,
        representation=args.representation,
        decimal_places=resolve_decimal_places(),
    )

    prompt_template = load_prompt_template(args.prompt)
    prompt = prompt_template.format(
        question=args.question,
        summary=summary,
        data=data_text,
    )

    if args.show_prompt:
        print("\n--- Prompt ---\n")
        print(prompt)
        print("\n--- End Prompt ---\n")

    backend = OllamaBackend()
    response_text = backend.generate(model=args.model, prompt=prompt)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(response_text)


if __name__ == "__main__":
    main()
