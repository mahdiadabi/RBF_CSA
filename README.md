# DDoS paper implementations

Two independent Python implementations:

1. `singh_mlp_ga.py`: 5-3-1 sigmoid MLP whose 18 weights are encoded as a 288-bit chromosome and optimized by a genetic algorithm.
2. `rbf_csa.py`: NSL-KDD preprocessing, GA feature selection, and a Gaussian RBF classifier trained with Cuckoo Search.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

## Singh 2017

Input CSV columns:

- `incoming_ip_count`
- `request_count`
- `constant_port_mapping` (0/1)
- `fixed_frame_length` (0/1)
- `label` (0 normal, 1 attack)

```bash
python -m ddos_papers.singh_mlp_ga data/singh.csv --seed 42
```

The paper does not publish the GA population size, crossover/mutation rates, split, seed, or classification threshold. They are command-line parameters in this implementation.

## GA-RBF-CSA

Input is a headerless NSL-KDD CSV with the standard 41 features plus `label` and optional `difficulty`.

```bash
python -m ddos_papers.rbf_csa data/KDDTrain+.txt --seed 42
```

Only `normal` and DoS records are retained. Categorical columns are one-hot encoded before GA selection. The paper's statement that GA selects 9 of 41 features conflicts with categorical encoding; this implementation groups encoded columns by original feature and selects exactly 9 original features.

The paper omits the RBF hidden-unit count, epsilon, alpha, Levy sampler details, seed, and categorical encoding. These are explicit configurable defaults here.

## Tests

```bash
pytest -q
```
# RBF_CSA
