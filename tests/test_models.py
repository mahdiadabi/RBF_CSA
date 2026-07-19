import numpy as np
import pandas as pd
from ddos_papers.singh_mlp_ga import BinaryGAMLP, SinghGAConfig, engineer_features
from ddos_papers.rbf_csa import RBFClassifier, CuckooSearch, CSAConfig


def test_singh_shapes_and_training():
    frame = pd.DataFrame({
        "incoming_ip_count": [2, 3, 25, 30] * 5,
        "request_count": [20, 40, 4000, 5000] * 5,
        "constant_port_mapping": [0, 0, 1, 1] * 5,
        "fixed_frame_length": [0, 0, 1, 1] * 5,
    })
    x = engineer_features(frame)
    y = np.array([0, 0, 1, 1] * 5)
    model = BinaryGAMLP(SinghGAConfig(population_size=20, generations=20, seed=1)).fit(x, y)
    assert x.shape == (20, 5)
    assert model.predict_proba(x).shape == (20,)
    assert model.best_fitness_ > 0.8


def test_rbf_csa_reduces_to_valid_scores():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, (30, 2))
    y = (x[:, 0] > 0).astype(float)
    model = RBFClassifier(input_size=2, hidden_units=3)
    search = CuckooSearch(model, CSAConfig(population_size=10, iterations=10, seed=2)).fit(x, y)
    scores = model.scores(search.best_solution_, x)
    assert scores.shape == (30,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert np.isfinite(search.best_mse_)


def test_rbf_csa_batched_losses_match_individual_scores():
    rng = np.random.default_rng(3)
    inputs = rng.uniform(-1, 1, (12, 2))
    targets = (inputs[:, 0] > 0).astype(float)
    model = RBFClassifier(input_size=2, hidden_units=3)
    search = CuckooSearch(
        model,
        CSAConfig(population_size=4, loss_batch_size=2, seed=3),
    )
    nests = rng.uniform(-1, 1, (4, model.solution_size))

    expected = np.array(
        [
            np.mean((model.scores(nest, inputs) - targets) ** 2)
            for nest in nests
        ]
    )

    assert np.allclose(search._losses(nests, inputs, targets), expected)
