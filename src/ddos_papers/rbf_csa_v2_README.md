# `rbf_csa_v2.py`

`rbf_csa_v2.py` is a command-line implementation of a binary NSL-KDD DDoS
detector. It retains `normal` and DoS records, labels DoS records as `1`, and
uses the following training pipeline:

1. Split the retained records into stratified training (80%) and test (20%)
   sets.
2. Impute and scale numeric features, then one-hot encode categorical features.
3. Use a genetic algorithm (GA) to select exactly nine of the 41 original
   NSL-KDD feature groups.
4. Initialize a Gaussian RBF classifier with K-means centers and a logistic
   regression output layer.
5. Optimize the RBF parameters with Cuckoo Search using Lévy flights.
6. Choose the F1-maximizing decision threshold on a validation split from the
   training data and report metrics.

## Requirements

Install the project and its dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

The input must be a headerless NSL-KDD file with the standard 41 feature
columns followed by `label`, with an optional final `difficulty` column.

## Run

```bash
python -m ddos_papers.rbf_csa_v2 data/KDDTrain+.txt \
  --seed 42 \
  --hidden-units 10 \
  --ga-iterations 100 \
  --csa-iterations 300
```

Arguments:

- `nsl_kdd`: required path to the NSL-KDD CSV file.
- `--seed`: random seed used by data splitting, GA, K-means, and Cuckoo Search
  (default: `42`).
- `--hidden-units`: Gaussian RBF hidden units (default: `10`).
- `--ga-iterations`: feature-selection GA iterations (default: `100`).
- `--csa-iterations`: Cuckoo Search iterations (default: `300`).

Progress is written to standard error. Standard output contains an indented JSON
document with the selected original features, GA/CSA losses, the selected
threshold, classification and probability metrics, a confusion matrix, and a
classification report for the training, test, and complete retained datasets.

## Reproducibility Notes

Feature selection operates on groups derived from the original 41 features, so
one-hot-encoded categorical columns are selected together. The GA always
selects nine groups. Cuckoo Search minimizes a weighted combination of log
loss, `1 - F1`, `1 - attack recall`, and L2 penalties for RBF centers, spreads,
and output weights. Model and search defaults are defined in `CSAConfig` and
`FeatureGAConfig` in `rbf_csa_v2.py`.
