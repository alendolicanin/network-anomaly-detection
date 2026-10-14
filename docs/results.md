# Reference results

| Contents | Location |
| --- | --- |
| Selected L8, AE p98 and IF p95 | `results/decision.json` |
| Development metrics and ranking | `results/development/` |
| Final 2017 test | `results/summary_final_test/` |
| External 2018 test | `results/external/summary_external_test/` |
| February 14 diagnostics | `results/external/summary_external_diagnostic/` |
| Epochs, model sizes and durations | `results/development/training_runs.csv`, `results/training_main.json`, `results/training_external.json` |
| Latent activity, residuals and class score distributions | `results/diagnostics_main/`, `results/diagnostics_external/` |
| Test figures | `results/figures_final_test/`, `results/external/figures_external_test/` |

Paths are relative to the repository root. `per_run.csv` records individual runs; `mean_sd.csv` records their means and sample standard deviations. `per_class.csv` and `per_day.csv` group results by attack label and source daily member. CSV files retain full precision and use decimal points.

At the selected L8/p98 operating point, final-test F1 is 0.7321 ± 0.0480 for 2017 and 0.2307 ± 0.0342 for the external 2018 test. These are mean ± sample SD across five seeds. Other operating points are retained in the CSV files.

The final-test summaries evaluate 26 fitted models: L8, L16, L32, Isolation Forest and the RobustScaler branch across five seeds, plus one PCA. Development includes the other architectures, giving 41 distinct fits in the main branch. The external branch has 20 fits, reused for the external test and supplementary diagnostics. Evaluation counts therefore differ from training counts.

Numerical results and the decision retain their original values. Model hashes in the decision refer to the full experiment archive; its fitted models and per-flow scores are not included here. Summary `integrity.json` files and the release manifest do not replace those omitted artifacts.
