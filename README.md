# TS-SymboLLM

[![CI](https://github.com/noahthebuzz/TS-SymboLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/noahthebuzz/TS-SymboLLM/actions/workflows/ci.yml)

**Does compressing a time series into a compact symbolic string help a large
language model interpret it better than handing it raw or rounded numbers?**
That's the question this project exists to answer. TS-SymboLLM is a
local-first tool for interpreting time-series data with a local LLM (via
[Ollama](https://ollama.com)), comparing three representations of the same
series — raw values, rounded values, and PAA/SAX symbolic strings — against
your own data or a bundled example.

This repository lets you:

- Interpret a single dataset with a local Ollama model, choosing the representation.
- Run a `benchmark` sweep across models × representations × datasets, with structured, analyzable output.
- Try it without your own data via an optional synthetic-data showcase.

## Quickstart

1. Install and start [Ollama](https://ollama.com) (a local LLM runtime), then pull a model:

```bash
ollama pull qwen2.5:7b
```

2. Clone this repository and install the package:

```bash
pip install -e .
```

3. Run a bundled example, comparing representations:

```bash
ts-symbollm --example temperature --model qwen2.5:7b --representation raw
ts-symbollm --example temperature --model qwen2.5:7b --representation symbolic
```

List the available example datasets:

```bash
ts-symbollm --list-examples
```

Don't want to install the package? `pip install -r requirements.txt` and
run `python app.py ...` instead of `ts-symbollm ...` — same behavior,
just without the installed command.

## Use your own data

Provide a CSV or JSON file with time series data:

```bash
ts-symbollm --data /path/to/your/data.csv --model qwen2.5:7b
```

`--model` accepts any model name you've already pulled in your local
Ollama install — it isn't limited to the bundled examples' default.

### Supported data formats

#### CSV (single series)

```csv
timestamp,value
2024-03-01T08:00:00,48.2
2024-03-01T08:05:00,48.9
```

#### CSV (multiple series)

Include a `series` (or `name`) column to group values:

```csv
timestamp,series,value
2024-03-01T08:00:00,cpu_utilization,32.5
2024-03-01T08:00:00,cpu_temperature,46.1
```

#### JSON (single series)

```json
{
  "data": {
    "2024-03-01T08:00:00": 48.2,
    "2024-03-01T08:05:00": 48.9
  }
}
```

#### JSON (multiple series)

```json
{
  "series": [
    {
      "name": "cpu_utilization_percent",
      "data": {
        "2024-03-01T08:00:00": 32.5,
        "2024-03-01T08:05:00": 35.2
      }
    },
    {
      "name": "cpu_temperature_c",
      "data": {
        "2024-03-01T08:00:00": 46.1,
        "2024-03-01T08:05:00": 46.8
      }
    }
  ]
}
```

## Choosing a representation

`--representation raw|rounded|symbolic` controls how series values are
rendered in the prompt (default `raw`):

- **`raw`** — the actual numeric values, unchanged.
- **`rounded`** — rounded to a configurable number of decimal places
  (`representation.rounded_decimal_places` in config, default whole numbers).
- **`symbolic`** — compresses the series via Piecewise Aggregate
  Approximation (PAA) and Symbolic Aggregate approXimation (SAX) into a
  compact letter string (e.g. `fbaaacdefg...`) instead of listing numeric
  values. Two parameters control the compression, each overridable via a
  CLI flag or the config file's `representation` section:
  - `--alphabet-size` / `representation.symbolic_levels` — the SAX alphabet size (default `10`).
  - `--paa-segments` / `representation.symbolic_segment_length` — how many raw points collapse into one symbol by default (default `10`, i.e. roughly `n/10` symbols for a series of length `n`).

Every run automatically saves a diagram of the input series — in
whichever representation was used — to `./plots/`, no extra flag needed.
Override the location with `--plot-dir` or the config file's
`plotting.output_dir`, or skip it entirely with `--no-plot`.

## Prompt customization

Prompt templates are plain text files with placeholders:

- `{question}` — your analysis request
- `{summary}` — auto-generated series stats (always reflects the real underlying values, even in symbolic mode)
- `{data}` — the series data, formatted per the chosen representation

Example:

```bash
ts-symbollm --example cpu \
  --prompt examples/prompts/interpretation.txt \
  --question "Explain the relationship between utilization and temperature."
```

## Useful CLI options

```bash
ts-symbollm --help
```

- `--max-points`: limit how many points per series are included in the prompt for `raw`/`rounded` (defaults to 120; not needed for `symbolic`, which is already compressed).
- `--show-prompt`: print the final prompt before sending it to the model.
- `--output`: write the model response to a file.

## Benchmarking

`ts-symbollm benchmark` runs a model × representation × dataset matrix in
one non-interactive command and writes one structured row per run —
this is the tool built specifically to answer the research question
above at scale, rather than one dataset at a time:

```bash
ts-symbollm benchmark \
  --data temperature,cpu \
  --model qwen2.5:7b,mistral:7b \
  --representation raw,rounded,symbolic \
  --results ./results/results.jsonl
```

- `--data`: comma-separated dataset paths and/or bundled example names — plug in any of your own CSV/JSON files here, mixed freely with bundled examples.
- `--model`: comma-separated model names to sweep — any model already pulled in your local Ollama install.
- `--representation`: comma-separated `raw`/`rounded`/`symbolic` values to sweep (default `raw`).
- `--params`: comma-separated Ollama sampling-parameter presets to sweep: `default`, `rational`, `creative` (default `default`).
- `--results`: path to the structured results file — JSON Lines or CSV, inferred from the extension (default `./results/results.jsonl`). Loadable with `pandas.read_json(path, lines=True)` or `csv.DictReader`.
- `--score`: enable automatic scoring for datasets with known ground truth (currently just the bundled `temperature` example — see `ts_symbollm/scoring.py` to add more, or a different kind of scorer entirely).
- `--paa-segments`, `--alphabet-size`, `--plot-dir`, `--no-plot`: same meaning as the single-run flags above; a diagram is still saved automatically per dataset/representation combination (not per model or params preset).

Each combination in the matrix produces one row with the model, model
tier, representation, dataset, sampling params, prompt, response,
latency, and (if `--score` applies) a score and notes. A short summary
table prints at the end of the run.

Note: unlike the single-run flow, `benchmark` currently sends the full
series in `raw`/`rounded` mode with no `--max-points`-style truncation —
keep this in mind for very long datasets.

## Results

No published results yet — run `ts-symbollm benchmark` yourself against
your own data and models to see how raw, rounded, and symbolic
representations compare.

## Trying it without your own data

Don't have a dataset on hand? `examples/showcase/` is an optional script
that fabricates synthetic single- and multi-series datasets (simulated
CPU temperature, capacity, fan speed, etc.) plus diagrams and
raw/rounded/symbolic prompt files, purely so you have something to run the
tool against. It's a demo aid, not a core feature — the installable
`ts_symbollm` package has no dependency on it. See
[`examples/showcase/README.md`](examples/showcase/README.md) for usage.

## Model backends

The tool talks to models through a small `Backend` interface
(`ts_symbollm/backends/base.py`) rather than depending on Ollama directly
anywhere else in the codebase. `ts_symbollm/backends/ollama.py`'s
`OllamaBackend` is the only implementation today. To add another backend
(e.g. a hosted API for comparison), subclass `Backend`, implement
`generate(model, prompt, params) -> str` (and `pull(model) -> bool` if the
runtime needs an explicit download step), and add it alongside `ollama.py`
under `ts_symbollm/backends/`.

## Configuration

Shipped defaults (model tiers, Ollama sampling presets, representation
settings, the plot output directory) live in one file:
`ts_symbollm/config/config.json`. To override them without editing the
package, put your own JSON file at `~/.config/ts-symbollm/config.json`, or
point the `TS_SYMBOLLM_CONFIG` environment variable at any file — a
top-level section you provide (e.g. `"models"` or `"plotting"`) replaces
that section entirely rather than merging individual keys.

## Repository layout

```
.
├── app.py                  # Thin CLI entry point (same as running `ts-symbollm`)
├── pyproject.toml          # Installable package definition
├── examples/
│   ├── data/                # Example datasets
│   ├── prompts/             # Example prompt templates
│   └── showcase/            # Optional synthetic-data demo (not part of the package)
├── tests/                   # Unit tests
└── ts_symbollm/             # Installable package (CLI, benchmark runner, backends, config, representation, plotting)
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests
```

Tests don't require a running Ollama server (network calls are mocked)
and don't write any plot images outside of pytest's own temporary
directories. CI (GitHub Actions) runs the same suite on every push to
`master` and on every pull request.

## Context / Background

This project started as a bachelor thesis at the University of Ulm's
Institute for Database and Information Systems, exploring whether large
language models could support real-time monitoring in Digital Twin
systems — analyzing multivariate sensor time series (CPU temperature,
utilization, fan speed, voltage), diagnosing the likely cause of an
anomaly, and suggesting corrective actions, using a simulated
server-overheating scenario as the test case.

The core question carried over into this repository unchanged: since
LLMs are fundamentally text-based, does a time series compressed into a
compact symbolic string (via PAA/SAX) get interpreted more reliably than
the same series handed over as long sequences of raw or rounded numbers?
The thesis compared several locally-run open models of different sizes
across raw, rounded, and symbolic representations, under different
sampling configurations, and graded responses on diagnostic accuracy, the
quality of suggested corrective actions, and even the coherence of
generated flowchart diagrams. The short version of what came out of it:
symbolic aggregation genuinely helped, especially as the data grew larger
or more multivariate, and it let smaller, cheaper models hold their own
against much larger ones — though compressing too aggressively could
throw away detail that mattered, and larger models still tended to edge
out smaller ones on raw data.

The thesis version of this project was built for that one academic
evaluation: synthetic datasets, one-off scripts, a manually graded
evaluation sheet, and code with a specific person's hardware and username
baked in. This repository has since been reworked into a general-purpose,
installable tool — `ts_symbollm`'s config, representation, prompt,
backend, and results modules, plus the `benchmark` subcommand — so the
same raw-vs-symbolic comparison can be run by anyone, against their own
time-series data and local models, rather than staying a one-time
research artifact.

## Notes

- This project expects a **local** Ollama server to be running.
- For large datasets, use `--max-points` (single-run mode) to keep prompts small.
- The example files are intentionally compact for quick testing.
