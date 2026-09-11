from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Optional

from .config import config


@dataclass
class ResultRecord:
    '''
    One structured row of benchmark output -- the source of truth for a
    run, superseding the old free-text log files under logs/{prompt_type}/
    {model_size}/{model_name}/{params}/{data_representation}/*.log
    (`logger.log()` still writes those, as an optional secondary output).
    '''

    timestamp: str
    model: str
    model_size: str
    representation: str
    dataset_id: str
    params: Optional[dict]
    prompt: str
    response: str
    latency_seconds: float
    score: Optional[float] = None
    score_notes: Optional[str] = None

    @classmethod
    def create(
        cls,
        model: str,
        representation: str,
        dataset_id: str,
        params: Optional[dict],
        prompt: str,
        response: str,
        latency_seconds: float,
        model_size: Optional[str] = None,
        timestamp: Optional[str] = None,
        score: Optional[float] = None,
        score_notes: Optional[str] = None,
    ) -> "ResultRecord":
        '''
        Convenience constructor: resolves model_size/timestamp if not
        given. `score`/`score_notes` are optional -- scoring (see
        scoring.py) is an opt-in step; a run with no scorer just leaves
        these as None.
        '''
        return cls(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            model=model,
            model_size=model_size or config.resolve_model_size(model),
            representation=representation,
            dataset_id=dataset_id,
            params=params,
            prompt=prompt,
            response=response,
            latency_seconds=latency_seconds,
            score=score,
            score_notes=score_notes,
        )


FIELD_NAMES = [f.name for f in fields(ResultRecord)]


class ResultsWriter:
    '''
    Appends one `ResultRecord` per benchmark run to a results file, in
    JSON Lines (one JSON object per line -- loadable with
    `pandas.read_json(path, lines=True)`) or CSV (loadable with
    `csv.DictReader`). Format is inferred from the file extension
    (".csv" -> csv, anything else -> jsonl) unless given explicitly.
    '''

    def __init__(self, path: str, format: Optional[str] = None):
        self.path = path
        self.format = format or self._infer_format(path)
        if self.format not in ("jsonl", "csv"):
            raise ValueError(f"Unsupported results format: {self.format!r} (use 'jsonl' or 'csv')")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _infer_format(path: str) -> str:
        return "csv" if os.path.splitext(path)[1].lower() == ".csv" else "jsonl"

    def write(self, record: ResultRecord) -> None:
        if self.format == "csv":
            self._write_csv(record)
        else:
            self._write_jsonl(record)

    def _write_jsonl(self, record: ResultRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record)) + "\n")

    def _write_csv(self, record: ResultRecord) -> None:
        row = asdict(record)
        row["params"] = json.dumps(row["params"]) if row["params"] is not None else ""
        file_is_empty = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELD_NAMES)
            if file_is_empty:
                writer.writeheader()
            writer.writerow(row)
