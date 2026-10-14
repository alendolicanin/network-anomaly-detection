# Network Anomaly Detection with Autoencoders

Code and reference results for Alen Dolićanin's master's thesis on anomaly detection in network traffic. The experiments compare reconstruction-based autoencoders with Isolation Forest and PCA on CICIDS2017 Improved, with a separate replication on two daily members of CSE-CIC-IDS2018 Improved.

The public-data pipeline trains on normal traffic, calibrates percentile thresholds on a separate normal partition, and evaluates a fixed development decision on held-out data. A smaller Keras example in `synthetic/` demonstrates the method with five synthetic traffic features.

## Reference results

Development selects the L8 autoencoder and the 98th-percentile threshold. Its test F1 is **0.7321 ± 0.0480** on CICIDS2017 Improved and **0.2307 ± 0.0342** on the external test. Values are mean ± sample standard deviation across five seeds. The external experiment uses new training and calibration; it does not transfer the fitted 2017 model.

See [the experimental setup](docs/experimental_setup.md) for partitions, label policies and selection rules, and [the result index](docs/results.md) for the corresponding CSV files.

## Repository contents

| Path | Contents |
| --- | --- |
| `ids/` | Data preparation, training, calibration, evaluation and plotting |
| `configs/` | Experiment settings, data sources and expected checksums |
| `scripts/` | Download, analysis and integrity checks |
| `tests/` | Unit tests and a small integration experiment |
| `results/` | Reference metrics, diagnostics, decisions and figures |
| `evidence/` | Code fingerprints, data identities and feature schemas |
| `synthetic/` | Synthetic data, Keras notebooks and reference figures |
| `docs/` | Experimental setup, implementation notes and result index |
| `experiment_workflow.ipynb` | Notebook interface to the public-data pipeline |

Raw public datasets, prepared matrices, fitted models and per-flow scores are not included. The supplied summaries can be inspected without training. Full reproduction downloads the source data and writes new outputs under `work/`.

## Installation

The public-data experiments were run on Linux with Python 3.12.14. Exact dependency versions are pinned in `requirements.txt`. TensorFlow is needed only for the separate synthetic example.

Run the following from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows Command Prompt, activate the environment with `.venv\Scripts\activate.bat` instead of `source .venv/bin/activate`.

The commands below use this active environment. See [implementation notes](docs/implementation.md) for the recorded execution environment and the limits of the synthetic environment record.

## Verify the supplied package

```bash
python scripts/verify_release.py
python scripts/check_saved_results.py
python -m unittest discover -s tests -v
```

- `verify_release.py` checks the file sizes, SHA-256 checksums and Python syntax recorded for this release. Run it before making local edits.
- `check_saved_results.py` checks the distributed code and configuration, their connection to the recorded experiment, saved decision, result checksums and mean/SD calculations.
- The integration test uses small generated fixtures, two training epochs and three Isolation Forest trees. It checks pipeline behavior, not the thesis results.

These checks do not reconstruct predictions from the omitted fitted models or per-flow scores.

## Download and prepare data

Allow at least 25 GB for downloads and preparation, plus additional space for models and experiment outputs. Preparation uses several GB of RAM; these are planning estimates, not measured minimum requirements.

```bash
python scripts/download_data.py --dataset cic2017 --root work
python scripts/download_data.py --dataset cic2018 --root work
python -m ids prepare --source-root work/data_sources --dataset cic2017 --output work/prepared2017
python -m ids prepare --source-root work/data_sources --dataset cic2018 --output work/prepared2018
python -m ids verify-data --data work/prepared2017
python -m ids verify-data --data work/prepared2018
```

Source URLs and expected hashes are in [`configs/sources.json`](configs/sources.json). The 2018 downloader uses HTTP Range requests to extract two daily CSV members. If the server does not return the expected response, download the archive from the configured URL and extract `Wednesday-14-02-2018.csv` and `Thursday-15-02-2018.csv` into `work/data_sources/improved2018/`. Then rerun the download command to verify them. The expected hashes identify the required source files and should not be changed to accept a different release.

## Run the experiments

Use a fresh experiment directory and execute these commands in order. Development analysis must precede the final and external stages.

```bash
python -m ids develop --data work/prepared2017 --experiment work/experiment
python scripts/analyze_development.py --code . --experiment work/experiment --output work/development_summary
python -m ids final --data work/prepared2017 --experiment work/experiment
python -m ids external --data work/prepared2018 --experiment work/experiment
```

`develop` trains six autoencoder architectures and Isolation Forest across five seeds, then locks the selected architecture and percentiles. `final` evaluates that decision and adds PCA and a RobustScaler analysis. `external` retrains L16, L32, the selected architecture and Isolation Forest on the 2018 normal partition. The reference run contains 61 distinct fits: 41 in the main branch and 20 in the external branch.

Export summaries, figures and diagnostics:

```bash
python -m ids summarize --experiment work/experiment
python -m ids figures --experiment work/experiment
python -m ids summarize --experiment work/experiment --external
python -m ids figures --experiment work/experiment --external
python -m ids summarize --experiment work/experiment --external --diagnostic
python -m ids figures --experiment work/experiment --external --diagnostic
python scripts/diagnostics.py --experiment work/experiment --data work/prepared2017 --output work/diagnostics_main
python scripts/diagnostics.py --experiment work/experiment --data work/prepared2018 --output work/diagnostics_external --external
```

### Expected runtime

The saved records contain about **144 minutes of fitting and training monitoring** for all 61 distinct public-data fits: approximately 33 minutes for development, 8 minutes for PCA and RobustScaler, and 104 minutes for the external branch. These rounded stage totals come from `results/development/training_runs.csv`, `results/training_main.json` and `results/training_external.json`; overlapping development models are counted once.

This is a measured training subtotal on the recorded CPUs, not an end-to-end benchmark. Allow several hours for a full reproduction on comparable hardware, plus dataset download time. Preparation, calibration, evaluation, diagnostics and exports add time. CPU speed, RAM, disk throughput and network access affect the total.

### Resuming a run

Start a fresh experiment with this distribution and keep its code and configuration fixed throughout the run. The saved reference records retain their original execution fingerprints; see [provenance](evidence/README.md). Completed fits are reused only after integrity checks; an interrupted fit restarts at its first epoch. After the decision is locked, continue with `final` and `external`. Preparation and summary/figure exports require fresh output directories.

Do not copy `results/decision.json` into a new experiment: each reproduction derives its own decision from its development outputs.

### Notebook interface

Optionally install JupyterLab in the same environment and open `experiment_workflow.ipynb` from the repository root. Execute cells in order. Training cells run the full experiments; pilot outputs only check the execution path.

## Synthetic example

Follow [`synthetic/README.md`](synthetic/README.md) in a separate environment. The notebooks generate 1,750 examples and train four distinct Keras autoencoder architectures. Their reference results and environment history are separate from the public-data experiments.

## Interpretation

The public datasets represent laboratory traffic. The protocol was formulated after earlier results from the same datasets had been inspected, and the new chronological split does not remove that limitation. Five seeds measure training variability on fixed populations, not uncertainty across five independent networks. Further methodological details are in [the experimental setup](docs/experimental_setup.md).
