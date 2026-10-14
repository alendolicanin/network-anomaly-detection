# Implementation

The public-data pipeline is available through `python -m ids`.

| Module | Responsibility |
| --- | --- |
| `ids/data.py` | Source validation, partitions, feature encoding and scaling |
| `ids/models.py` | Model fitting and anomaly scores |
| `ids/metrics.py` | Percentile thresholds and evaluation metrics |
| `ids/pipeline.py` | Development, final evaluation and external replication |
| `ids/report.py` | Summary tables and figures |
| `ids/core.py` | Configuration, fingerprints and shared integrity checks |

`development` selects the architecture and percentile. `final_test` evaluates the locked decision. `external_test` uses fresh training and calibration in the external domain. `external_diagnostic` separately describes attacks and Attempted flows from the day used for normal training. Test fixtures are not experimental results.

## Recorded environment

The public-data records report Python 3.12.14 and the versions pinned in `requirements.txt`, running on Linux 6.18.44, x86_64, glibc 2.39, with nine available logical CPUs and 10,451,464,192 bytes of RAM. Training runs used AMD EPYC 9V74 and Intel Xeon Platinum 8573C processors. Numerical libraries and Isolation Forest were restricted to one thread. No GPU was used. This records execution resources, not minimum system requirements.

The synthetic notebooks record Python 3.10.0 in their metadata. Exact historical Jupyter, TensorFlow, Keras and other library versions, operating system and hardware are not established by the available records. The public-data environment must not be attributed to that earlier example. See [the synthetic instructions](../synthetic/README.md).

## Data provenance

Source URLs, expected hashes and schemas are in [`configs/sources.json`](../configs/sources.json).

| Source | Recorded retrieval time (UTC) |
| --- | --- |
| `CICIDS2017_improved.zip` | 2026-10-05 21:08:44 |
| `Wednesday-14-02-2018.csv` | 2026-10-05 21:12:10 |
| `Thursday-15-02-2018.csv` | 2026-10-05 21:12:27 |

For 2018, checksums verify the extracted daily members; a checksum for the complete remote archive was not recorded. [`evidence/README.md`](../evidence/README.md) explains the retained data and code identities.

## Figures and language

Original experimental figures are stored under `results/`; synthetic reference figures are under `synthetic/reference_figures/`. Figure labels and some runtime messages remain in Serbian to match the thesis. Documentation and repository filenames use English.
