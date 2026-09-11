from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import ollama

# examples/ ships at the repo root, not inside the package, so this only
# resolves for an editable install (`pip install -e .`) run from a checkout.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = {
    "temperature": os.path.join(REPO_ROOT, "examples", "data", "temperature.csv"),
    "cpu": os.path.join(REPO_ROOT, "examples", "data", "cpu_multi.json"),
}

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


def _format_value(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


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


def format_series_data(series_data: Dict[str, List[Tuple[str, float]]], max_points: int | None) -> str:
    lines = []
    for name, points in series_data.items():
        lines.append(f"{name}:")
        for timestamp, value in _downsample(points, max_points):
            lines.append(f"  {timestamp}: {_format_value(value)}")
    return "\n".join(lines)


def load_prompt_template(path: str | None) -> str:
    if not path:
        return DEFAULT_PROMPT_TEMPLATE
    with open(path, encoding="utf-8") as file:
        return file.read()


def generate_with_ollama(model: str, prompt: str) -> str:
    response_text = ""
    response = ollama.generate(model=model, prompt=prompt, stream=True)
    for part in response:
        chunk = part.get("response", "")
        print(chunk, end="", flush=True)
        response_text += chunk
    print()
    return response_text


def list_examples() -> None:
    print("Available examples:")
    for name, path in EXAMPLES.items():
        print(f"  - {name}: {os.path.relpath(path, REPO_ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze time series data with a local Ollama model.",
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_examples:
        list_examples()
        return

    if bool(args.data) == bool(args.example):
        parser.error("Provide either --data or --example (but not both).")

    data_path = EXAMPLES.get(args.example) if args.example else args.data
    if not data_path:
        parser.error("No data path resolved. Use --data or --example.")

    series_data = load_time_series(data_path)
    summary = summarize_series(series_data)
    data_text = format_series_data(series_data, args.max_points)

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

    response_text = generate_with_ollama(args.model, prompt)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(response_text)


if __name__ == "__main__":
    main()
