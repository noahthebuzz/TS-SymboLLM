# TS-SymboLLM

Local LLM showcase for interpreting time-series data. This repository lets you:

- Run a **local Ollama model** against your own CSV/JSON time series.
- Try **built-in example datasets** without any extra setup.
- Customize prompts for different analysis tasks (anomalies, summaries, forecasts, etc.).
- Generate **synthetic demo data** if you don't have a dataset handy yet (see [Trying it without your own data](#trying-it-without-your-own-data)).

## Quickstart

1. Install and start Ollama (local LLM runtime).
2. Pull a model (example uses `qwen2.5:7b`):

```bash
ollama pull qwen2.5:7b
```

3. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run an example:

```bash
python app.py --example temperature --model qwen2.5:7b
```

List the available example datasets:

```bash
python app.py --list-examples
```

5. Try the symbolic representation — this is the actual research question
   behind this tool: does compressing a series into a compact SAX string
   help an LLM interpret it better than raw or rounded numbers?

```bash
python app.py --example temperature --model qwen2.5:7b --representation symbolic
```

## Use your own data

Provide a CSV or JSON file with time series data:

```bash
python app.py --data /path/to/your/data.csv --model qwen2.5:7b
```

Add a custom prompt template:

```bash
python app.py --data /path/to/your/data.json \
  --prompt examples/prompts/interpretation.txt \
  --question "Summarize trends and highlight anomalies."
```

## Trying it without your own data

Don't have a dataset on hand? `examples/showcase/` is an optional script
that fabricates synthetic single- and multi-series datasets (simulated
CPU temperature, capacity, fan speed, etc.) plus diagrams and
raw/rounded/symbolic prompt files, purely so you have something to run the
tool against. It's a demo aid, not a core feature — the installable
`ts_symbollm` package has no dependency on it. See
[`examples/showcase/README.md`](examples/showcase/README.md) for usage.

## Supported data formats

### CSV (single series)

```csv
timestamp,value
2024-03-01T08:00:00,48.2
2024-03-01T08:05:00,48.9
```

### CSV (multiple series)

Include a `series` (or `name`) column to group values:

```csv
timestamp,series,value
2024-03-01T08:00:00,cpu_utilization,32.5
2024-03-01T08:00:00,cpu_temperature,46.1
```

### JSON (single series)

```json
{
  "data": {
    "2024-03-01T08:00:00": 48.2,
    "2024-03-01T08:05:00": 48.9
  }
}
```

### JSON (multiple series)

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

## Prompt customization

Prompt templates are plain text files with placeholders:

- `{question}` — your analysis request
- `{summary}` — auto-generated series stats
- `{data}` — sampled data points

Example:

```bash
python app.py --example cpu \
  --prompt examples/prompts/interpretation.txt \
  --question "Explain the relationship between utilization and temperature."
```

## Useful CLI options

```bash
python app.py --help
```

- `--max-points`: limit how many points per series are included in the prompt (defaults to 120).
- `--show-prompt`: print the final prompt before sending it to the model.
- `--output`: write the model response to a file.
- `--representation raw|rounded|symbolic`: how series values are rendered in the prompt (default `raw`). `symbolic` compresses each series into a compact SAX letter string (e.g. `fbaaacdefg...`) instead of listing numeric values.
- `--paa-segments`: number of PAA segments for `--representation symbolic` (default: `ceil(n/10)`, overridable via config).
- `--alphabet-size`: SAX alphabet size for `--representation symbolic` (default: from config).
- `--plot-dir`: where to save the diagram generated for this run (default `./plots/`, also overridable via config — see below).
- `--no-plot`: skip generating a diagram for this run.

Every run automatically saves a diagram of the input series to `./plots/`
(a single figure, with all series overlaid and a legend for multi-series
data) — no extra flag needed.

## Configuration

Shipped defaults (model tiers, Ollama sampling presets, representation
settings, the plot output directory) live in
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
└── ts_symbollm/             # Installable package (CLI, config, representation, plotting, legacy harness)
```

## Installing as a package

Instead of `python app.py ...`, you can also install the tool and use the
`ts-symbollm` command from any directory:

```bash
pip install -e .
ts-symbollm --example temperature --model qwen2.5:7b
```

## Notes

- This project expects a **local** Ollama server to be running.
- For large datasets, use `--max-points` to keep prompts small.
- The example files are intentionally compact for quick testing.
