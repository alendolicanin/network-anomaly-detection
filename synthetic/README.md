# Synthetic traffic example

This example uses five numerical traffic features, with 1,500 normal examples, 125 SYN flood examples and 125 port scan examples. The generated CSV is included in `notebooks/`.

## Run the notebooks

Create and activate a separate Python environment, then run from the repository root:

```bash
python -m pip install -r synthetic/requirements.txt
cd synthetic/notebooks
python -m jupyterlab
```

Execute `01_generate_data.ipynb`, then `02_train_autoencoder.ipynb`. The kernel working directory must be `synthetic/notebooks/`. The first notebook regenerates the supplied dataset. The second trains four distinct autoencoders and writes tables, predictions and figures to its working directory.

## Experimental setup

Normal examples are split into 1,050 training, 225 validation and 225 test examples. All 250 attack examples enter the test set. The scaler is fitted only on normal training data.

Model 1, Model 3 and Latent 2 reuse the same trained base autoencoder. Model 3 applies a different threshold percentile. Evaluation variants are not separate training runs.

## Reference outputs and environment

Notebook outputs and `reference_figures/` retain the earlier execution results. The notebook metadata records Python 3.10.0, but does not establish the exact historical Jupyter, TensorFlow, Keras or other library versions, operating system, CPU, RAM or GPU use. The public-data execution environment does not describe this example.

`requirements.txt` retains the dependency constraints from the previous package; it is not a complete lockfile for the original execution. Identical numerical results in another environment are not guaranteed. Plot labels and saved runtime output remain in Serbian to match the thesis.
