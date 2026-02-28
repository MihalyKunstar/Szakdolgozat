# Hospital Contact Network Prototype (Mesa, no infection)

Single-file Python prototype in `main.py` for generating a one-day hospital contact network (agent-based) without SIR/SEIR transmission.

## Environment setup (Windows / VS Code)

### venv
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### conda (optional)
```bash
conda create -n hospital-proto python=3.11 -y
conda activate hospital-proto
pip install -r requirements.txt
```

## Run
```bash
python main.py --seed 42
```
Optional:
```bash
python main.py --seed 42 --run_id 1001
```

## Outputs
After run, files are saved under `outputs/`:
- `visit_log.csv`
- `aggregated_edges.csv`
- `run_summary.csv`
- `figures/network.png`
- `figures/timeseries.png`
- `figures/degree_hist.png`
