# Showcase: synthetic data generator

This is an **optional demo**, not part of the installable `ts_symbollm`
package — a way to try `ts-symbollm` without bringing your own time-series
data. It fabricates synthetic single- and multi-series datasets (simulated
CPU temperature, capacity, fan speed, voltage, room temperature) along with
diagrams and raw/rounded/symbolic prompt JSON files.

## Usage

Run it directly from a checkout with `ts_symbollm` installed
(`pip install -e .` from the repo root):

```bash
python examples/showcase/generate.py
```

By default this generates the temperature dataset. Edit the
`generate_data(...)` call at the bottom of `generate.py` to enable the
other datasets (`random`, `sequence`, `cpu_temp_cap_tsd`, `multi_3_tsd`,
`multi_4_tsd`, `multi_5_tsd`).

Output goes to `prompts/` (raw/rounded/symbolic JSON files, one per
representation) and each representation's configured plot directory
(diagrams, via `ts_symbollm.plotting`).

## Files

- `datagen.py` — the synthetic series generators themselves (pure data,
  no plotting).
- `generate.py` — orchestrates generation, plotting (via
  `ts_symbollm.plotting.plot_series`), and writing prompt JSON files.
