from __future__ import annotations
import argparse, json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .metrics import classification_metrics


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def engineer_features(frame: pd.DataFrame) -> np.ndarray:
    required = {"incoming_ip_count", "request_count", "constant_port_mapping", "fixed_frame_length"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return np.column_stack([
        (frame["incoming_ip_count"].to_numpy() >= 20).astype(float),
        (frame["request_count"].to_numpy() >= 3500).astype(float),
        frame["constant_port_mapping"].to_numpy(dtype=float),
        frame["fixed_frame_length"].to_numpy(dtype=float),
        -np.ones(len(frame)),
    ])


@dataclass
class SinghGAConfig:
    population_size: int = 100
    generations: int = 2000
    crossover_rate: float = 0.8
    mutation_rate: float = 1.0 / 288.0
    tournament_size: int = 3
    target_fitness: float = 0.99
    seed: int = 42


class BinaryGAMLP:
    input_units = 5
    hidden_units = 3
    output_units = 1
    bits_per_weight = 16
    weight_count = 18
    chromosome_length = 288

    def __init__(self, config: SinghGAConfig | None = None):
        self.config = config or SinghGAConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.best_chromosome_: np.ndarray | None = None
        self.best_fitness_: float | None = None
        self.generations_run_: int = 0

    @staticmethod
    def decode(chromosomes: np.ndarray) -> np.ndarray:
        shaped = chromosomes.reshape(len(chromosomes), 18, 16)
        fractions = np.power(2.0, -np.arange(1, 17))
        return (shaped @ fractions) * 20.0 - 10.0

    @classmethod
    def forward_many(cls, chromosomes: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        weights = cls.decode(chromosomes)
        input_hidden = weights[:, :15].reshape(-1, 5, 3)
        hidden_output = weights[:, 15:].reshape(-1, 3, 1)
        hidden = sigmoid(np.einsum("ni,pij->pnj", inputs, input_hidden))
        return sigmoid(np.einsum("pnj,pjk->pnk", hidden, hidden_output))[:, :, 0]

    def _fitness(self, population: np.ndarray, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        outputs = self.forward_many(population, inputs)
        sample_error = 0.5 * (targets[None, :] - outputs) ** 2
        return 1.0 - sample_error.sum(axis=1) / len(targets)

    def _select(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        candidates = self.rng.integers(0, len(population), size=(len(population), self.config.tournament_size))
        winners = candidates[np.arange(len(population)), np.argmax(fitness[candidates], axis=1)]
        return population[winners].copy()

    def _reproduce(self, selected: np.ndarray) -> np.ndarray:
        offspring = selected.copy()
        self.rng.shuffle(offspring, axis=0)
        for index in range(0, len(offspring) - 1, 2):
            if self.rng.random() < self.config.crossover_rate:
                point = self.rng.integers(1, self.chromosome_length)
                tail = offspring[index, point:].copy()
                offspring[index, point:] = offspring[index + 1, point:]
                offspring[index + 1, point:] = tail
        mutations = self.rng.random(offspring.shape) < self.config.mutation_rate
        return np.logical_xor(offspring, mutations).astype(np.uint8)

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> "BinaryGAMLP":
        population = self.rng.integers(0, 2, (self.config.population_size, self.chromosome_length), dtype=np.uint8)
        best_fitness = -np.inf
        best = population[0].copy()
        for generation in range(1, self.config.generations + 1):
            fitness = self._fitness(population, inputs, targets)
            index = int(np.argmax(fitness))
            if fitness[index] > best_fitness:
                best_fitness, best = float(fitness[index]), population[index].copy()
            self.generations_run_ = generation
            if best_fitness >= self.config.target_fitness:
                break
            population = self._reproduce(self._select(population, fitness))
            population[0] = best
        self.best_chromosome_, self.best_fitness_ = best, best_fitness
        return self

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        if self.best_chromosome_ is None:
            raise RuntimeError("Model is not fitted")
        return self.forward_many(self.best_chromosome_[None, :], inputs)[0]

    def predict(self, inputs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(inputs) >= threshold).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    inputs, targets = engineer_features(frame), frame["label"].to_numpy(dtype=float)
    train_x, test_x, train_y, test_y = train_test_split(inputs, targets, test_size=0.2, stratify=targets, random_state=args.seed)
    config = SinghGAConfig(population_size=args.population, generations=args.generations, seed=args.seed)
    model = BinaryGAMLP(config).fit(train_x, train_y)
    result = classification_metrics(test_y.astype(int), model.predict(test_x, args.threshold))
    result.update({"fitness": model.best_fitness_, "generations": model.generations_run_})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
