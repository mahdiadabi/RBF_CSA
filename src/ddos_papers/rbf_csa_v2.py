from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .metrics import classification_metrics, regression_metrics
from .nsl_kdd import (
    FEATURES,
    build_preprocessor,
    original_feature_groups,
    read_binary_nsl_kdd,
)


def report_progress(
    stage: str,
    current: int,
    total: int,
    *,
    best_loss: float | None = None,
    start_time: float | None = None,
) -> None:
    percentage = 100 * current / max(total, 1)
    message = f"{stage}: {current}/{total} ({percentage:5.1f}%)"
    if best_loss is not None:
        message += f", best loss={best_loss:.6f}"
    if start_time is not None and current > 0:
        elapsed = time.monotonic() - start_time
        remaining = elapsed * (total - current) / current
        message += f", ETA={remaining:.0f}s"
    print(message, file=sys.stderr, flush=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


class RBFClassifier:
    def __init__(self, input_size: int, hidden_units: int):
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.solution_size = hidden_units * (input_size + 2) + 1

    def unpack(
        self,
        solution: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        center_end = self.hidden_units * self.input_size
        centers = solution[:center_end].reshape(self.hidden_units, self.input_size)

        spread_start = center_end
        spread_end = spread_start + self.hidden_units
        spreads = np.maximum(solution[spread_start:spread_end], 1e-4)

        weight_start = spread_end
        weight_end = weight_start + self.hidden_units
        weights = solution[weight_start:weight_end]

        bias = float(solution[-1])
        return centers, spreads, weights, bias

    def basis(
        self,
        solution: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        centers, spreads, _, _ = self.unpack(solution)
        squared = np.sum(
            (inputs[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        return np.exp(-squared / (2.0 * spreads[None, :] ** 2))

    def logits(
        self,
        solution: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        basis = self.basis(solution, inputs)
        _, _, weights, bias = self.unpack(solution)
        return basis @ weights + bias

    def scores(
        self,
        solution: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        return sigmoid(self.logits(solution, inputs))


@dataclass(slots=True)
class CSAConfig:
    population_size: int = 50
    iterations: int = 300
    discovery_probability: float = 0.25
    alpha: float = 1.0
    levy_beta: float = 1.5
    epsilon: float = 1e-6
    seed: int = 42
    weight_clip: float = 5.0
    bias_clip: float = 5.0
    center_clip: float = 3.0
    spread_min: float = 1e-3
    spread_max: float = 3.0
    l2_weight: float = 1e-4
    l2_center: float = 1e-5
    l2_spread: float = 1e-5
    alpha_decay: float = 0.995


class CuckooSearch:
    def __init__(self, model: RBFClassifier, config: CSAConfig | None = None):
        self.model = model
        self.config = config or CSAConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.best_solution_: np.ndarray | None = None
        self.best_loss_: float = np.inf

    def _spread_slice(self) -> slice:
        center_end = self.model.hidden_units * self.model.input_size
        return slice(center_end, center_end + self.model.hidden_units)

    def _weight_slice(self) -> slice:
        center_end = self.model.hidden_units * self.model.input_size
        spread_end = center_end + self.model.hidden_units
        return slice(spread_end, spread_end + self.model.hidden_units)

    def _apply_bounds(self, nests: np.ndarray) -> np.ndarray:
        spread_slice = self._spread_slice()
        weight_slice = self._weight_slice()

        nests[:, : spread_slice.start] = np.clip(
            nests[:, : spread_slice.start],
            -self.config.center_clip,
            self.config.center_clip,
        )
        nests[:, spread_slice] = np.clip(
            nests[:, spread_slice],
            self.config.spread_min,
            self.config.spread_max,
        )
        nests[:, weight_slice] = np.clip(
            nests[:, weight_slice],
            -self.config.weight_clip,
            self.config.weight_clip,
        )
        nests[:, -1] = np.clip(
            nests[:, -1],
            -self.config.bias_clip,
            self.config.bias_clip,
        )
        return nests

    def _levy(self, size: tuple[int, int]) -> np.ndarray:
        beta = self.config.levy_beta

        sigma_u = (
            math.gamma(1.0 + beta)
            * math.sin(math.pi * beta / 2.0)
            / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
        ) ** (1.0 / beta)

        numerator = self.rng.normal(0.0, sigma_u, size=size)
        denominator = self.rng.normal(0.0, 1.0, size=size)

        return numerator / np.maximum(
            np.abs(denominator) ** (1.0 / beta),
            1e-12,
        )

    def _objective_terms(
        self,
        solution: np.ndarray,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, float, float, float]:
        probs = np.clip(self.model.scores(solution, inputs), 1e-7, 1 - 1e-7)
        preds = (probs >= 0.5).astype(int)

        ll = log_loss(targets, probs, labels=[0, 1])
        f1 = f1_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, pos_label=1, zero_division=0)

        centers, spreads, weights, _ = self.model.unpack(solution)
        reg = (
            self.config.l2_weight * float(np.sum(weights**2))
            + self.config.l2_center * float(np.sum(centers**2))
            + self.config.l2_spread * float(np.sum(spreads**2))
        )

        loss = 0.5 * ll + 0.3 * (1.0 - f1) + 0.2 * (1.0 - rec) + reg
        return loss, ll, f1, rec

    def _losses(
        self,
        nests: np.ndarray,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            [self._objective_terms(nest, inputs, targets)[0] for nest in nests],
            dtype=float,
        )

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        initial_solution: np.ndarray | None = None,
    ) -> "CuckooSearch":
        pop = self.config.population_size
        dim = self.model.solution_size

        nests = self.rng.uniform(-1.0, 1.0, size=(pop, dim))

        spread_slice = self._spread_slice()
        nests[:, spread_slice] = self.rng.uniform(
            self.config.spread_min,
            min(1.0, self.config.spread_max),
            size=(pop, self.model.hidden_units),
        )

        if initial_solution is not None:
            nests[0] = initial_solution.copy()

        nests = self._apply_bounds(nests)
        losses = self._losses(nests, inputs, targets)

        alpha = self.config.alpha
        start_time = time.monotonic()
        report_every = max(1, self.config.iterations // 10)

        for iteration in range(self.config.iterations):
            elite_idx = int(np.argmin(losses))
            elite = nests[elite_idx].copy()
            elite_loss = float(losses[elite_idx])

            if elite_loss < self.best_loss_:
                self.best_loss_ = elite_loss
                self.best_solution_ = elite.copy()

            if self.best_loss_ <= self.config.epsilon:
                report_progress(
                    "CSA",
                    iteration + 1,
                    self.config.iterations,
                    best_loss=self.best_loss_,
                    start_time=start_time,
                )
                break

            steps = self._levy((pop, dim))
            candidates = nests + alpha * steps * (nests - elite)
            candidates = self._apply_bounds(candidates)

            candidate_losses = self._losses(candidates, inputs, targets)
            improved = candidate_losses < losses
            nests[improved] = candidates[improved]
            losses[improved] = candidate_losses[improved]

            abandon_mask = self.rng.random(pop) < self.config.discovery_probability
            if np.any(abandon_mask):
                idx = np.flatnonzero(abandon_mask)
                a = nests[self.rng.integers(0, pop, size=idx.size)]
                b = nests[self.rng.integers(0, pop, size=idx.size)]
                trial = nests[idx] + self.rng.random((idx.size, 1)) * (a - b)
                trial = self._apply_bounds(trial)
                trial_losses = self._losses(trial, inputs, targets)
                improved_local = trial_losses < losses[idx]
                nests[idx[improved_local]] = trial[improved_local]
                losses[idx[improved_local]] = trial_losses[improved_local]

            alpha *= self.config.alpha_decay
            if (
                (iteration + 1) % report_every == 0
                or iteration + 1 == self.config.iterations
            ):
                report_progress(
                    "CSA",
                    iteration + 1,
                    self.config.iterations,
                    best_loss=self.best_loss_,
                    start_time=start_time,
                )

        elite_idx = int(np.argmin(losses))
        elite = nests[elite_idx].copy()
        elite_loss = float(losses[elite_idx])

        if elite_loss < self.best_loss_ or self.best_solution_ is None:
            self.best_loss_ = elite_loss
            self.best_solution_ = elite

        return self


@dataclass(slots=True)
class FeatureGAConfig:
    population_size: int = 20
    iterations: int = 100
    crossover_probability: float = 0.5
    mutation_probability: float = 0.2
    tournament_size: int = 3
    selected_count: int = 9
    seed: int = 42


class GroupedFeatureGA:
    def __init__(
        self,
        groups: list[np.ndarray],
        hidden_units: int,
        config: FeatureGAConfig | None = None,
    ):
        self.groups = groups
        self.hidden_units = hidden_units
        self.config = config or FeatureGAConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.best_mask_: np.ndarray | None = None
        self.best_loss_: float = np.inf

    def _repair(self, mask: np.ndarray) -> np.ndarray:
        selected = np.flatnonzero(mask)
        target = self.config.selected_count

        if selected.size > target:
            off = self.rng.choice(selected, size=selected.size - target, replace=False)
            mask[off] = False
        elif selected.size < target:
            unselected = np.flatnonzero(~mask)
            on = self.rng.choice(
                unselected,
                size=target - selected.size,
                replace=False,
            )
            mask[on] = True
        return mask

    def _columns(self, mask: np.ndarray) -> np.ndarray:
        return np.concatenate([self.groups[i] for i in np.flatnonzero(mask)])

    def _fitness(
        self,
        mask: np.ndarray,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
    ) -> float:
        columns = self._columns(mask)
        x_train = train_x[:, columns]
        x_val = val_x[:, columns]

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.config.seed,
        )
        clf.fit(x_train, train_y)
        probs = np.clip(clf.predict_proba(x_val)[:, 1], 1e-7, 1 - 1e-7)
        preds = (probs >= 0.5).astype(int)

        ll = log_loss(val_y, probs, labels=[0, 1])
        f1 = f1_score(val_y, preds, zero_division=0)
        rec = recall_score(val_y, preds, pos_label=1, zero_division=0)
        return 0.5 * ll + 0.3 * (1.0 - f1) + 0.2 * (1.0 - rec)

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> "GroupedFeatureGA":
        n_groups = len(self.groups)
        population = self.rng.integers(
            0,
            2,
            size=(self.config.population_size, n_groups),
            dtype=bool,
        )
        population = np.array(
            [self._repair(row.copy()) for row in population], dtype=bool
        )

        train_x, val_x, train_y, val_y = train_test_split(
            inputs,
            targets,
            test_size=0.2,
            random_state=self.config.seed,
            stratify=targets,
        )

        cache: dict[bytes, float] = {}
        start_time = time.monotonic()
        report_every = max(1, self.config.iterations // 10)

        def evaluate_mask(mask: np.ndarray) -> float:
            key = mask.tobytes()
            if key not in cache:
                cache[key] = self._fitness(mask, train_x, train_y, val_x, val_y)
            return cache[key]

        for iteration in range(self.config.iterations):
            losses = np.array([evaluate_mask(row) for row in population], dtype=float)
            elite_idx = int(np.argmin(losses))
            elite = population[elite_idx].copy()
            elite_loss = float(losses[elite_idx])

            if elite_loss < self.best_loss_:
                self.best_loss_ = elite_loss
                self.best_mask_ = elite.copy()

            children: list[np.ndarray] = [elite.copy()]

            while len(children) < self.config.population_size:

                def tournament() -> np.ndarray:
                    idx = self.rng.choice(
                        self.config.population_size,
                        size=self.config.tournament_size,
                        replace=False,
                    )
                    winner = idx[np.argmin(losses[idx])]
                    return population[winner].copy()

                parent_a = tournament()
                parent_b = tournament()

                child_a = parent_a.copy()
                child_b = parent_b.copy()

                if (
                    self.rng.random() < self.config.crossover_probability
                    and n_groups > 1
                ):
                    point = self.rng.integers(1, n_groups)
                    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
                    child_b = np.concatenate([parent_b[:point], parent_a[point:]])

                for child in (child_a, child_b):
                    mutation_mask = (
                        self.rng.random(n_groups) < self.config.mutation_probability
                    )
                    child[mutation_mask] = ~child[mutation_mask]
                    child = self._repair(child)
                    children.append(child.copy())
                    if len(children) >= self.config.population_size:
                        break

            population = np.array(children[: self.config.population_size], dtype=bool)
            if (
                (iteration + 1) % report_every == 0
                or iteration + 1 == self.config.iterations
            ):
                report_progress(
                    "Feature GA",
                    iteration + 1,
                    self.config.iterations,
                    best_loss=self.best_loss_,
                    start_time=start_time,
                )

        final_losses = np.array([evaluate_mask(row) for row in population], dtype=float)
        elite_idx = int(np.argmin(final_losses))
        elite = population[elite_idx].copy()
        elite_loss = float(final_losses[elite_idx])

        if elite_loss < self.best_loss_ or self.best_mask_ is None:
            self.best_loss_ = elite_loss
            self.best_mask_ = elite

        return self

    def selected_columns(self) -> np.ndarray:
        if self.best_mask_ is None:
            raise RuntimeError("Feature GA has not been fitted.")
        return self._columns(self.best_mask_)


def build_kmeans_initial_solution(
    model: RBFClassifier,
    train_x: np.ndarray,
    train_y: np.ndarray,
    seed: int,
) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=model.hidden_units,
        random_state=seed,
        n_init=10,
    )
    kmeans.fit(train_x)
    centers = kmeans.cluster_centers_

    pairwise = np.sqrt(
        np.maximum(
            np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2),
            0.0,
        )
    )
    nonzero = pairwise[pairwise > 0]
    if nonzero.size == 0:
        global_spread = 1.0
    else:
        global_spread = float(np.mean(nonzero))
    spreads = np.full(
        model.hidden_units, max(global_spread / np.sqrt(2 * model.hidden_units), 1e-3)
    )

    squared = np.sum((train_x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    basis = np.exp(-squared / (2.0 * spreads[None, :] ** 2))

    output = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=seed,
    )
    output.fit(basis, train_y)

    weights = output.coef_.ravel()
    bias = float(output.intercept_[0])

    return np.concatenate(
        [
            centers.reshape(-1),
            spreads,
            weights,
            np.array([bias], dtype=float),
        ]
    )


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if thresholds.size == 0:
        return 0.5

    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx])


def evaluate_with_threshold(
    model: RBFClassifier,
    solution: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    probs = np.clip(model.scores(solution, x), 1e-7, 1 - 1e-7)
    preds = (probs >= threshold).astype(int)

    reg = regression_metrics(y, probs)
    cls = classification_metrics(y, preds)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result: dict[str, object] = {}
    result.update(reg)
    result.update(cls)
    result["threshold"] = threshold
    result["log_loss"] = float(log_loss(y, probs, labels=[0, 1]))
    result["f1"] = float(f1_score(y, preds, zero_division=0))
    result["recall_attack"] = float(
        recall_score(y, preds, pos_label=1, zero_division=0)
    )
    result["balanced_accuracy"] = float(balanced_accuracy_score(y, preds))
    result["roc_auc"] = float(roc_auc_score(y, probs))
    result["pr_auc"] = float(average_precision_score(y, probs))
    result["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    result["classification_report"] = classification_report(
        y,
        preds,
        output_dict=True,
        zero_division=0,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary NSL-KDD DDoS detector using GA-selected features and RBF+CSA.",
    )
    parser.add_argument("nsl_kdd", type=Path, help="Path to KDDTrain+.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-units", type=int, default=10)
    parser.add_argument("--ga-iterations", type=int, default=100)
    parser.add_argument("--csa-iterations", type=int, default=300)
    args = parser.parse_args()

    print("Loading NSL-KDD data...", file=sys.stderr, flush=True)
    frame, labels = read_binary_nsl_kdd(args.nsl_kdd)

    train_frame, test_frame, train_y, test_y = train_test_split(
        frame,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    preprocessor = build_preprocessor(train_frame)
    print("Preprocessing features...", file=sys.stderr, flush=True)
    train_x = preprocessor.fit_transform(train_frame)
    test_x = preprocessor.transform(test_frame)
    whole_x = preprocessor.transform(frame)
    whole_y = labels

    groups = original_feature_groups(preprocessor)

    ga = GroupedFeatureGA(
        groups=groups,
        hidden_units=args.hidden_units,
        config=FeatureGAConfig(
            iterations=args.ga_iterations,
            seed=args.seed,
        ),
    )
    print("Selecting feature groups...", file=sys.stderr, flush=True)
    ga.fit(train_x, train_y)
    selected_columns = ga.selected_columns()

    selected_original_features = [
        FEATURES[index] for index, chosen in enumerate(ga.best_mask_) if chosen
    ]

    final_train_x, val_x, final_train_y, val_y = train_test_split(
        train_x[:, selected_columns],
        train_y,
        test_size=0.2,
        random_state=args.seed,
        stratify=train_y,
    )

    model = RBFClassifier(
        input_size=final_train_x.shape[1],
        hidden_units=args.hidden_units,
    )

    initial_solution = build_kmeans_initial_solution(
        model=model,
        train_x=final_train_x,
        train_y=final_train_y,
        seed=args.seed,
    )

    print("Optimizing RBF classifier...", file=sys.stderr, flush=True)
    csa = CuckooSearch(
        model=model,
        config=CSAConfig(
            iterations=args.csa_iterations,
            seed=args.seed,
        ),
    )
    csa.fit(final_train_x, final_train_y, initial_solution=initial_solution)

    if csa.best_solution_ is None:
        raise RuntimeError("CSA failed to produce a final solution.")

    val_probs = np.clip(model.scores(csa.best_solution_, val_x), 1e-7, 1 - 1e-7)
    print("Evaluating final model...", file=sys.stderr, flush=True)
    best_threshold = find_best_threshold(val_y, val_probs)

    result = {
        "feature_ga_best_loss": ga.best_loss_,
        "final_csa_best_loss": csa.best_loss_,
        "selected_original_features": selected_original_features,
        "training_set": evaluate_with_threshold(
            model,
            csa.best_solution_,
            train_x[:, selected_columns],
            train_y,
            best_threshold,
        ),
        "test_set": evaluate_with_threshold(
            model,
            csa.best_solution_,
            test_x[:, selected_columns],
            test_y,
            best_threshold,
        ),
        "whole_set": evaluate_with_threshold(
            model,
            csa.best_solution_,
            whole_x[:, selected_columns],
            whole_y,
            best_threshold,
        ),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
