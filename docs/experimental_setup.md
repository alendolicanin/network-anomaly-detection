# Experimental setup

The experiments use CICIDS2017 Improved and two complete daily members of CSE-CIC-IDS2018 Improved. Source URLs, checksums, expected columns and flow counts are recorded in [`configs/sources.json`](../configs/sources.json).

## Data partitions

Normal BENIGN flows are ordered by UTC timestamp and ID. The chronological split is approximately 60% training, 20% validation and 20% calibration, with boundaries moved to the end of each equal-timestamp group.

For 2017, Monday supplies normal training, validation and calibration; Tuesday and Wednesday form development; Thursday and Friday form the final test. For 2018, the February 14 member supplies normal training, validation and calibration, and the February 15 member supplies the external test. Member dates need not match every flow timestamp after UTC midnight.

| Partition | 2017 | 2018 |
| --- | ---: | ---: |
| Normal training | 222,972 | 3,366,478 |
| Normal validation | 74,324 | 1,122,160 |
| Normal calibration | 74,325 | 1,122,160 |
| Development | 818,718 | — |
| Final / external test | 909,632 | 5,410,088 |
| Additional diagnostics | — | 287,551 |

Both datasets yield 82 inputs: 78 numerical features and four protocol indicators. Constants, protocol categories and scaling parameters are determined from the corresponding training partition only. Unseen protocol categories produce zeros in those indicators. StandardScaler is the primary scaler; RobustScaler is a supplementary analysis.

## Label policies

The primary policy treats BENIGN and Attempted flows as negative, and effective attacks as positive. A supplementary policy excludes Attempted flows during evaluation without refitting or changing the threshold. Attempted flows never enter normal training, validation or calibration.

The February 14 diagnostic partition contains 193,354 FTP Attempted flows and 94,197 SSH attack flows, with no BENIGN flows. It is not an independent future test.

## Models and calibration

Autoencoders use `MLPRegressor` with ReLU hidden layers, a linear output and Adam. One `partial_fit` call corresponds to one epoch. Training allows at most 200 epochs, with patience 8 and minimum improvement `1e-5`. Normal validation reconstruction MSE controls early stopping, and the complete best accepted model state is retained. The six architectures and Isolation Forest parameters are specified in [`configs/experiment.json`](../configs/experiment.json). PCA is fitted once, deterministically.

Prepared matrices and scaler parameters use float64; scaled model input uses float32; MSE and percentile calibration use float64. An alarm is raised strictly when `score > threshold`. Thresholds are linear percentiles of normal calibration scores.

## Development decision

Architecture A* is selected at development p95 by highest mean F1, with ties resolved by lower FPR and fewer parameters. A common percentile is then selected for A* across the five seeds; Isolation Forest has a separate percentile selection. Percentile ties are resolved by lower FPR, then higher percentile. Final and external tests do not change this decision.

The reference decision selects L8, AE p98 and Isolation Forest p95.

## Aggregation and limitations

Seeds are 7, 21, 42, 84 and 123. Means and sample standard deviations are calculated across runs with `ddof=1`. PCA has no standard deviation. Undefined metrics are `null` in JSON and empty in CSV. Five seeds do not represent five independent network populations.

The protocol was formulated after earlier results from the same public datasets had been inspected. A new chronological split does not remove this limitation. File identity checks do not establish PCAP corpus completeness, absence of content duplicates or perfect labels.
