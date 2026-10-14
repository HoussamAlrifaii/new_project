# Run BikePulse yourself in VS Code

VS Code is an editor, not a requirement for training. Python runs the code from a terminal; VS Code provides a convenient terminal, notebook viewer and debugger. The cloud environment has already trained the model. You can reproduce it locally.

## 1. Open the project

Install Python 3.12 and VS Code. Install Microsoft's Python and Jupyter extensions. Clone the GitHub repository once published, then open its folder in VS Code. Do not create a second nested project folder.

## 2. Create an environment and install

In Terminal → New Terminal, run:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If local PowerShell policy blocks activation, use `.venv\Scripts\python.exe` directly instead of changing system policy. Then install:

```bash
python -m pip install -r requirements.txt
```

Use Python: Select Interpreter in VS Code and choose the project's `.venv`. In a notebook, choose the same environment with Select Kernel. The checked-in default interpreter path targets Linux/macOS; select the Windows executable manually on Windows.

## 3. Train and inspect results

```bash
python -m bikepulse.train
```

The first run downloads authentic UCI data. It engineers timestamp-aligned lags, fits both candidates, selects using chronological validation, refits the selected model, and evaluates the held-out period. It prints metrics and saves `artifacts/metrics.json`, `artifacts/predictions.csv` and `artifacts/model.pkl`. This command performs training, not just inference. Internet access to archive.ics.uci.edu is required for the first download.

For constrained Linux/macOS machines, limit training threads:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m bikepulse.train
```

## 4. Run the deeper EDA

Open `notebooks/01_eda_and_modeling.ipynb`, select the `.venv` kernel, and use Run All. Or execute reproducibly from the terminal:

```bash
python scripts/execute_notebook.py
```

The notebook includes hourly coverage, demand distributions, weekday/weekend patterns, month-by-hour heatmaps, weather associations, model comparisons, prediction diagnostics and largest errors. Charts have different purposes; additional charts alone do not make the project more rigorous. Explain what each reveals and where the evidence is limited.

## 5. Test and run the app

```bash
python -m pytest -q
python -m streamlit run app.py
```

Open the browser address printed by Streamlit on your own computer. The cloud onboarding interface does not expose a localhost preview. App tests use real data and may require the initial download. The first app run trains and caches its model.

## What to demonstrate to a recruiter

Show the problem definition, a few informative EDA charts, leakage prevention, chronological train/validation/test periods, baseline comparison, error analysis, and working interface. Explain a failure case as well as the average improvement. Do not claim present-day forecasting, live station availability, or a public deployment until those capabilities exist.

Next phase: public Streamlit deployment, rolling-origin validation and recent real data. A project can support a strong application but does not guarantee an interview.
