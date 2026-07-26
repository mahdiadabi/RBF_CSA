#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive MOGA-ACSA-RBF for Application Layer DDoS Detection

روش:
1. بارگذاری CSV
2. تبدیل label به دودویی
3. train/validation/test split
4. preprocessing بدون leakage
5. GA چندمعیاره برای انتخاب ویژگی
6. Adaptive CSA برای آموزش RBF
7. ارزیابی نهایی روی test

نمونه اجرا:
python adaptive_moga_acsa_rbf.py \
  --csv data.csv \
  --label-column label \
  --attack-label attack \
  --hidden-units 12 \
  --ga-population 24 \
  --ga-generations 40 \
  --csa-population 35 \
  --csa-iterations 150 \
  --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0 - eps))


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def print_progress(
    label: str,
    current: int,
    total: int,
    details: str = "",
    width: int = 30,
) -> None:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {details}" if details else ""
    end = "\n" if current >= total else ""

    sys.stdout.write(
        f"\r[{label}] [{bar}] {current}/{total} ({ratio * 100:5.1f}%){suffix}"
    )
    sys.stdout.write(end)
    sys.stdout.flush()


def classification_report_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_detection_rate": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_prob)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    try:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        result["roc_auc"] = float("nan")

    try:
        result["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        result["pr_auc"] = float("nan")

    return result


# ------------------------------------------------------------
# RBF Model
# ------------------------------------------------------------


class RBFClassifier:
    """
    Binary RBF classifier.

    solution vector:
    [
      centers: hidden_units * input_size,
      log_spreads: hidden_units,
      output_weights: hidden_units,
      bias: 1
    ]
    """

    def __init__(self, input_size: int, hidden_units: int):
        self.input_size = int(input_size)
        self.hidden_units = int(hidden_units)

        self.centers_size = self.hidden_units * self.input_size
        self.spreads_size = self.hidden_units
        self.weights_size = self.hidden_units
        self.bias_size = 1

        self.solution_size = (
            self.centers_size + self.spreads_size + self.weights_size + self.bias_size
        )

    def unpack(
        self, solution: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        solution = np.asarray(solution, dtype=float)

        idx = 0

        centers = solution[idx : idx + self.centers_size]
        centers = centers.reshape(self.hidden_units, self.input_size)
        idx += self.centers_size

        log_spreads = solution[idx : idx + self.spreads_size]
        spreads = np.exp(np.clip(log_spreads, -7.0, 3.0))
        spreads = np.clip(spreads, 1e-4, 20.0)
        idx += self.spreads_size

        weights = solution[idx : idx + self.weights_size]
        idx += self.weights_size

        bias = float(solution[idx])

        return centers, spreads, weights, bias

    def predict_proba(self, solution: np.ndarray, x: np.ndarray) -> np.ndarray:
        centers, spreads, weights, bias = self.unpack(solution)

        diff = x[:, None, :] - centers[None, :, :]
        squared_dist = np.sum(diff * diff, axis=2)

        phi = np.exp(-squared_dist / (2.0 * (spreads[None, :] ** 2)))
        logits = phi @ weights + bias

        return sigmoid(logits)


# ------------------------------------------------------------
# Cost-sensitive objective
# ------------------------------------------------------------


@dataclass
class CostSensitiveObjectiveConfig:
    lambda_bce: float = 0.50
    lambda_f1: float = 0.25
    lambda_fnr: float = 0.20
    lambda_l2: float = 0.05
    positive_class_weight: Optional[float] = None
    negative_class_weight: Optional[float] = None
    l2_scale: float = 1e-4
    threshold: float = 0.50


class CostSensitiveObjective:
    def __init__(self, config: CostSensitiveObjectiveConfig):
        self.config = config

    def _class_weights(self, y: np.ndarray) -> Tuple[float, float]:
        if (
            self.config.positive_class_weight is not None
            and self.config.negative_class_weight is not None
        ):
            return (
                float(self.config.negative_class_weight),
                float(self.config.positive_class_weight),
            )

        n = len(y)
        n_pos = max(int(np.sum(y == 1)), 1)
        n_neg = max(int(np.sum(y == 0)), 1)

        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)

        return float(w_neg), float(w_pos)

    def __call__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        solution: Optional[np.ndarray] = None,
    ) -> float:
        y_true = y_true.astype(int)
        y_prob = np.clip(y_prob, 1e-12, 1.0 - 1e-12)

        w_neg, w_pos = self._class_weights(y_true)

        bce = -np.mean(
            w_pos * y_true * safe_log(y_prob)
            + w_neg * (1 - y_true) * safe_log(1.0 - y_prob)
        )

        y_pred = (y_prob >= self.config.threshold).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fnr = fn / max(fn + tp, 1)

        if solution is None:
            l2 = 0.0
        else:
            l2 = self.config.l2_scale * float(np.mean(solution * solution))

        loss = (
            self.config.lambda_bce * bce
            + self.config.lambda_f1 * (1.0 - f1)
            + self.config.lambda_fnr * fnr
            + self.config.lambda_l2 * l2
        )

        return float(loss)


# ------------------------------------------------------------
# Adaptive Cuckoo Search
# ------------------------------------------------------------


@dataclass
class AdaptiveCSAConfig:
    population_size: int = 40
    iterations: int = 200

    alpha_max: float = 1.0
    alpha_min: float = 0.01

    pa_max: float = 0.35
    pa_min: float = 0.10

    levy_beta: float = 1.5

    center_bound: float = 3.0
    log_spread_min: float = -4.0
    log_spread_max: float = 1.5
    weight_bound: float = 5.0
    bias_bound: float = 5.0

    patience: int = 40
    min_delta: float = 1e-7

    seed: int = 42
    verbose: bool = False


class AdaptiveCuckooSearch:
    def __init__(
        self,
        model: RBFClassifier,
        objective: CostSensitiveObjective,
        config: AdaptiveCSAConfig,
    ):
        self.model = model
        self.objective = objective
        self.config = config
        self.rng = set_seed(config.seed)

        self.best_solution_: Optional[np.ndarray] = None
        self.best_loss_: float = float("inf")
        self.history_: List[float] = []

    def _levy(self, size: Tuple[int, ...]) -> np.ndarray:
        beta = self.config.levy_beta

        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = self.rng.normal(0, sigma_u, size)
        v = self.rng.normal(0, 1, size)

        return u / (np.abs(v) ** (1 / beta) + 1e-12)

    def _alpha(self, t: int) -> float:
        ratio = t / max(self.config.iterations - 1, 1)
        return self.config.alpha_min + (
            self.config.alpha_max - self.config.alpha_min
        ) * (1.0 - ratio)

    def _pa(self, t: int) -> float:
        ratio = t / max(self.config.iterations - 1, 1)
        return self.config.pa_max - (self.config.pa_max - self.config.pa_min) * ratio

    def _clip_solution(self, s: np.ndarray) -> np.ndarray:
        s = s.copy()

        idx = 0

        centers_end = idx + self.model.centers_size
        s[idx:centers_end] = np.clip(
            s[idx:centers_end],
            -self.config.center_bound,
            self.config.center_bound,
        )
        idx = centers_end

        spreads_end = idx + self.model.spreads_size
        s[idx:spreads_end] = np.clip(
            s[idx:spreads_end],
            self.config.log_spread_min,
            self.config.log_spread_max,
        )
        idx = spreads_end

        weights_end = idx + self.model.weights_size
        s[idx:weights_end] = np.clip(
            s[idx:weights_end],
            -self.config.weight_bound,
            self.config.weight_bound,
        )
        idx = weights_end

        s[idx] = np.clip(s[idx], -self.config.bias_bound, self.config.bias_bound)

        return s

    def _initial_population(self) -> np.ndarray:
        n = self.config.population_size
        d = self.model.solution_size

        pop = np.zeros((n, d), dtype=float)

        idx = 0

        centers_end = idx + self.model.centers_size
        pop[:, idx:centers_end] = self.rng.uniform(
            -1.0,
            1.0,
            size=(n, self.model.centers_size),
        )
        idx = centers_end

        spreads_end = idx + self.model.spreads_size
        pop[:, idx:spreads_end] = self.rng.uniform(
            -1.5,
            0.5,
            size=(n, self.model.spreads_size),
        )
        idx = spreads_end

        weights_end = idx + self.model.weights_size
        pop[:, idx:weights_end] = self.rng.uniform(
            -1.0,
            1.0,
            size=(n, self.model.weights_size),
        )
        idx = weights_end

        pop[:, idx] = self.rng.uniform(-0.5, 0.5, size=n)

        return np.array([self._clip_solution(s) for s in pop])

    def _losses(self, pop: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        losses = np.empty(pop.shape[0], dtype=float)

        for i, solution in enumerate(pop):
            y_prob = self.model.predict_proba(solution, x)
            losses[i] = self.objective(y, y_prob, solution)

        return losses

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pop = self._initial_population()
        losses = self._losses(pop, x, y)

        best_idx = int(np.argmin(losses))
        self.best_solution_ = pop[best_idx].copy()
        self.best_loss_ = float(losses[best_idx])

        no_improve = 0

        for t in range(self.config.iterations):
            alpha_t = self._alpha(t)
            pa_t = self._pa(t)

            current_best = pop[int(np.argmin(losses))].copy()

            levy_steps = self._levy(pop.shape)
            candidates = pop + alpha_t * levy_steps * (pop - current_best)
            candidates = np.array([self._clip_solution(s) for s in candidates])

            candidate_losses = self._losses(candidates, x, y)

            improve_mask = candidate_losses < losses
            pop[improve_mask] = candidates[improve_mask]
            losses[improve_mask] = candidate_losses[improve_mask]

            # Abandon worst nests
            abandon_mask = self.rng.random(self.config.population_size) < pa_t
            if np.any(abandon_mask):
                random_pop = self._initial_population()
                pop[abandon_mask] = random_pop[abandon_mask]
                losses[abandon_mask] = self._losses(pop[abandon_mask], x, y)

            # Local random walk
            order1 = self.rng.permutation(self.config.population_size)
            order2 = self.rng.permutation(self.config.population_size)
            step = self.rng.random(pop.shape) * (pop[order1] - pop[order2])
            local_candidates = pop + alpha_t * step
            local_candidates = np.array(
                [self._clip_solution(s) for s in local_candidates]
            )
            local_losses = self._losses(local_candidates, x, y)

            improve_mask = local_losses < losses
            pop[improve_mask] = local_candidates[improve_mask]
            losses[improve_mask] = local_losses[improve_mask]

            best_idx = int(np.argmin(losses))
            iter_best_loss = float(losses[best_idx])

            if iter_best_loss + self.config.min_delta < self.best_loss_:
                self.best_loss_ = iter_best_loss
                self.best_solution_ = pop[best_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            self.history_.append(self.best_loss_)

            if self.config.verbose:
                print_progress(
                    "CSA",
                    t + 1,
                    self.config.iterations,
                    details=f"loss={self.best_loss_:.6f}",
                )

            if no_improve >= self.config.patience:
                if self.config.verbose:
                    if t + 1 < self.config.iterations:
                        sys.stdout.write("\n")
                    print(f"[CSA] Early stopped at iteration {t}")
                break

        assert self.best_solution_ is not None
        return self.best_solution_


# ------------------------------------------------------------
# Multi-objective weighted GA for feature selection
# ------------------------------------------------------------


@dataclass
class FeatureGAConfig:
    population_size: int = 24
    generations: int = 40
    crossover_rate: float = 0.75
    mutation_rate: float = 0.08
    tournament_size: int = 3
    min_features: int = 3
    # max_features_ratio: float = 0.75
    max_features_ratio: float = 0.20
    alpha_f1: float = 0.60
    beta_fnr: float = 0.25
    # gamma_features: float = 0.15
    gamma_features: float = 0.40

    csa_population_size: int = 18
    csa_iterations: int = 50
    hidden_units: int = 8

    seed: int = 42
    verbose: bool = False


class MultiObjectiveFeatureGA:
    def __init__(
        self,
        config: FeatureGAConfig,
        objective_config: CostSensitiveObjectiveConfig,
    ):
        self.config = config
        self.objective_config = objective_config
        self.rng = set_seed(config.seed)

        self.best_mask_: Optional[np.ndarray] = None
        self.best_fitness_: float = float("inf")
        self.cache_: Dict[str, float] = {}
        self.history_: List[float] = []

    def _repair(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool).copy()
        n_features = len(mask)

        max_features = max(
            self.config.min_features,
            int(round(self.config.max_features_ratio * n_features)),
        )
        max_features = min(max_features, n_features)

        active = np.where(mask)[0]

        if len(active) < self.config.min_features:
            inactive = np.where(~mask)[0]
            need = self.config.min_features - len(active)
            add = self.rng.choice(
                inactive, size=min(need, len(inactive)), replace=False
            )
            mask[add] = True

        active = np.where(mask)[0]

        if len(active) > max_features:
            remove_count = len(active) - max_features
            remove = self.rng.choice(active, size=remove_count, replace=False)
            mask[remove] = False

        if not np.any(mask):
            mask[self.rng.integers(0, n_features)] = True

        return mask

    def _mask_key(self, mask: np.ndarray) -> str:
        return "".join("1" if b else "0" for b in mask)

    def _initial_population(self, n_features: int) -> np.ndarray:
        pop = self.rng.random((self.config.population_size, n_features)) < 0.5
        pop = np.array([self._repair(m) for m in pop], dtype=bool)
        return pop

    def _tournament(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        idxs = self.rng.choice(
            len(pop),
            size=self.config.tournament_size,
            replace=False,
        )
        best = idxs[int(np.argmin(fitness[idxs]))]
        return pop[best].copy()

    def _crossover(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.config.crossover_rate or len(p1) < 2:
            return p1.copy(), p2.copy()

        point = self.rng.integers(1, len(p1))
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])

        return self._repair(c1), self._repair(c2)

    def _mutate(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.copy()

        mutation = self.rng.random(len(mask)) < self.config.mutation_rate
        mask[mutation] = ~mask[mutation]

        return self._repair(mask)

    def _evaluate_mask(
        self,
        mask: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        key = self._mask_key(mask)
        if key in self.cache_:
            return self.cache_[key]

        selected = np.where(mask)[0]
        xtr = x_train[:, selected]
        xva = x_val[:, selected]

        model = RBFClassifier(
            input_size=xtr.shape[1],
            hidden_units=self.config.hidden_units,
        )

        objective = CostSensitiveObjective(self.objective_config)

        csa_config = AdaptiveCSAConfig(
            population_size=self.config.csa_population_size,
            iterations=self.config.csa_iterations,
            seed=self.config.seed + len(self.cache_),
            verbose=False,
            patience=max(10, self.config.csa_iterations // 3),
        )

        search = AdaptiveCuckooSearch(model, objective, csa_config)
        best_solution = search.fit(xtr, y_train)

        y_prob = model.predict_proba(best_solution, xva)
        y_pred = (y_prob >= self.objective_config.threshold).astype(int)

        f1 = f1_score(y_val, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        fnr = fn / max(fn + tp, 1)

        selected_ratio = len(selected) / x_train.shape[1]

        fitness = (
            self.config.alpha_f1 * (1.0 - f1)
            + self.config.beta_fnr * fnr
            + self.config.gamma_features * selected_ratio
        )

        fitness = float(fitness)
        self.cache_[key] = fitness

        return fitness

    def _evaluate_population(
        self,
        pop: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        label: str,
    ) -> np.ndarray:
        fitness = np.empty(len(pop), dtype=float)

        for index, mask in enumerate(pop):
            fitness[index] = self._evaluate_mask(
                mask,
                x_train,
                y_train,
                x_val,
                y_val,
            )

            if self.config.verbose:
                print_progress(
                    label,
                    index + 1,
                    len(pop),
                    details=f"best={np.min(fitness[: index + 1]):.6f}",
                )

        return fitness

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> np.ndarray:
        n_features = x_train.shape[1]

        pop = self._initial_population(n_features)
        fitness = self._evaluate_population(
            pop,
            x_train,
            y_train,
            x_val,
            y_val,
            label="GA init",
        )

        best_idx = int(np.argmin(fitness))
        self.best_mask_ = pop[best_idx].copy()
        self.best_fitness_ = float(fitness[best_idx])

        for gen in range(self.config.generations):
            new_pop = [self.best_mask_.copy()]  # elitism

            while len(new_pop) < self.config.population_size:
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)

                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                new_pop.append(c1)

                if len(new_pop) < self.config.population_size:
                    new_pop.append(c2)

            pop = np.array(new_pop, dtype=bool)

            fitness = self._evaluate_population(
                pop,
                x_train,
                y_train,
                x_val,
                y_val,
                label=f"GA {gen + 1}/{self.config.generations}",
            )

            best_idx = int(np.argmin(fitness))

            if fitness[best_idx] < self.best_fitness_:
                self.best_fitness_ = float(fitness[best_idx])
                self.best_mask_ = pop[best_idx].copy()

            self.history_.append(self.best_fitness_)

        assert self.best_mask_ is not None
        return self.best_mask_


# ------------------------------------------------------------
# Data loading and preprocessing
# ------------------------------------------------------------


def load_csv_binary(
    csv_path: str,
    label_column: str,
    attack_label: Optional[str] = None,
    normal_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    labels = df[label_column]
    x_df = df.drop(columns=[label_column])

    if attack_label is not None:
        y = (labels.astype(str) == str(attack_label)).astype(int).to_numpy()
    elif normal_label is not None:
        y = (labels.astype(str) != str(normal_label)).astype(int).to_numpy()
    else:
        unique = sorted(labels.astype(str).unique().tolist())

        if len(unique) != 2:
            raise ValueError(
                "Binary label conversion requires either --attack-label, "
                "--normal-label, or exactly two unique labels."
            )

        # حالت پیش‌فرض: label دوم کلاس حمله فرض می‌شود.
        y = (labels.astype(str) == unique[1]).astype(int).to_numpy()

    return x_df, y


def build_preprocessor_from_train(x_train_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x_train_df.columns if c not in numeric_cols]

    transformers = []

    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))

    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable columns found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def preprocess_splits(
    x_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], ColumnTransformer]:
    preprocessor = build_preprocessor_from_train(x_train_df)

    x_train = preprocessor.fit_transform(x_train_df)
    x_val = preprocessor.transform(x_val_df)
    x_test = preprocessor.transform(x_test_df)

    x_train = np.asarray(x_train, dtype=float)
    x_val = np.asarray(x_val, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(x_train.shape[1])]

    return x_train, x_val, x_test, feature_names, preprocessor


def load_selected_features(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        selected_features = data.get("selected_features")
    else:
        selected_features = data

    if not isinstance(selected_features, list) or not selected_features:
        raise ValueError(
            "Selected-features JSON must be a non-empty list or contain a "
            "'selected_features' list."
        )

    if not all(isinstance(name, str) and name for name in selected_features):
        raise ValueError("Every selected feature must be a non-empty string.")

    if len(selected_features) != len(set(selected_features)):
        raise ValueError("Selected-features JSON contains duplicate feature names.")

    return selected_features


def resolve_selected_feature_indices(
    selected_features: List[str],
    feature_names: List[str],
) -> List[int]:
    feature_indices = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in selected_features if name not in feature_indices]

    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Selected features are unavailable after preprocessing: "
            f"{missing_text}. Use the same dataset, split seed, and preprocessing."
        )

    return [feature_indices[name] for name in selected_features]


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------


@dataclass
class ExperimentConfig:
    csv: str
    label_column: str
    attack_label: Optional[str]
    normal_label: Optional[str]

    test_size: float = 0.20
    validation_size: float = 0.20

    hidden_units: int = 12

    ga_population: int = 24
    ga_generations: int = 40
    ga_mutation_rate: float = 0.08
    ga_crossover_rate: float = 0.75

    csa_population: int = 40
    csa_iterations: int = 200

    final_csa_population: int = 50
    final_csa_iterations: int = 300

    selected_features_json: Optional[str] = None
    seed: int = 42
    output: str = "results.json"
    verbose: bool = False


def run_experiment(config: ExperimentConfig) -> Dict:
    start_total = time.time()

    x_df, y = load_csv_binary(
        csv_path=config.csv,
        label_column=config.label_column,
        attack_label=config.attack_label,
        normal_label=config.normal_label,
    )

    x_trainval_df, x_test_df, y_trainval, y_test = train_test_split(
        x_df,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )

    val_ratio_inside_trainval = config.validation_size / (1.0 - config.test_size)

    x_train_df, x_val_df, y_train, y_val = train_test_split(
        x_trainval_df,
        y_trainval,
        test_size=val_ratio_inside_trainval,
        random_state=config.seed,
        stratify=y_trainval,
    )

    x_train, x_val, x_test, feature_names, _ = preprocess_splits(
        x_train_df,
        x_val_df,
        x_test_df,
    )

    if config.verbose:
        print("Shapes:")
        print("  train:", x_train.shape)
        print("  val:  ", x_val.shape)
        print("  test: ", x_test.shape)
        print("Class distribution:")
        print(
            "  train pos:", int(np.sum(y_train == 1)), "neg:", int(np.sum(y_train == 0))
        )
        print("  val   pos:", int(np.sum(y_val == 1)), "neg:", int(np.sum(y_val == 0)))
        print(
            "  test  pos:", int(np.sum(y_test == 1)), "neg:", int(np.sum(y_test == 0))
        )

    objective_config = CostSensitiveObjectiveConfig()

    ga_config = FeatureGAConfig(
        population_size=config.ga_population,
        generations=config.ga_generations,
        crossover_rate=config.ga_crossover_rate,
        mutation_rate=config.ga_mutation_rate,
        hidden_units=config.hidden_units,
        csa_population_size=config.csa_population,
        csa_iterations=config.csa_iterations,
        seed=config.seed,
        verbose=config.verbose,
    )

    if config.selected_features_json is not None:
        start_ga = time.time()
        selected_features = load_selected_features(config.selected_features_json)
        selected_idx = resolve_selected_feature_indices(
            selected_features,
            feature_names,
        )
        ga_time = time.time() - start_ga
        ga_best_fitness = None
        ga_history: List[float] = []
        feature_selection_mode = "loaded"

        if config.verbose:
            print(
                f"Loaded {len(selected_features)} selected features from "
                f"{config.selected_features_json}; GA skipped."
            )
    else:
        start_ga = time.time()
        ga = MultiObjectiveFeatureGA(
            config=ga_config,
            objective_config=objective_config,
        )
        selected_mask = ga.fit(x_train, y_train, x_val, y_val)
        selected_idx = np.where(selected_mask)[0].tolist()
        selected_features = [feature_names[i] for i in selected_idx]
        ga_time = time.time() - start_ga
        ga_best_fitness = float(ga.best_fitness_)
        ga_history = [float(value) for value in ga.history_]
        feature_selection_mode = "ga"

    x_train_final = np.vstack([x_train, x_val])[:, selected_idx]
    y_train_final = np.concatenate([y_train, y_val])

    x_test_selected = x_test[:, selected_idx]

    final_model = RBFClassifier(
        input_size=x_train_final.shape[1],
        hidden_units=config.hidden_units,
    )

    final_objective = CostSensitiveObjective(objective_config)

    final_csa_config = AdaptiveCSAConfig(
        population_size=config.final_csa_population,
        iterations=config.final_csa_iterations,
        seed=config.seed + 999,
        verbose=config.verbose,
        patience=max(30, config.final_csa_iterations // 4),
    )

    start_csa = time.time()

    final_search = AdaptiveCuckooSearch(
        model=final_model,
        objective=final_objective,
        config=final_csa_config,
    )

    best_solution = final_search.fit(x_train_final, y_train_final)

    csa_time = time.time() - start_csa

    y_train_prob = final_model.predict_proba(best_solution, x_train_final)
    y_test_prob = final_model.predict_proba(best_solution, x_test_selected)

    train_metrics = classification_report_binary(y_train_final, y_train_prob)
    test_metrics = classification_report_binary(y_test, y_test_prob)

    total_time = time.time() - start_total

    result = {
        "method": "CS-MOGA-ACSA-RBF",
        "description": (
            "Cost-sensitive RBF classifier trained by Adaptive Cuckoo Search "
            "with multi-objective GA feature selection."
        ),
        "config": asdict(config),
        "objective_config": asdict(objective_config),
        "ga_config": asdict(ga_config),
        "final_csa_config": asdict(final_csa_config),
        "data": {
            "raw_features": int(x_df.shape[1]),
            "encoded_features": int(x_train.shape[1]),
            "selected_features_count": int(len(selected_idx)),
            "selected_features_ratio": float(len(selected_idx) / x_train.shape[1]),
            "train_size": int(len(y_train)),
            "validation_size": int(len(y_val)),
            "test_size": int(len(y_test)),
            "train_positive": int(np.sum(y_train == 1)),
            "train_negative": int(np.sum(y_train == 0)),
            "test_positive": int(np.sum(y_test == 1)),
            "test_negative": int(np.sum(y_test == 0)),
        },
        "selected_features": selected_features,
        "feature_selection": {
            "mode": feature_selection_mode,
            "source_json": config.selected_features_json,
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "optimization": {
            "ga_best_fitness": ga_best_fitness,
            "final_csa_best_loss": float(final_search.best_loss_),
            "ga_history": ga_history,
            "final_csa_history": [float(v) for v in final_search.history_],
        },
        "runtime_seconds": {
            "ga": float(ga_time),
            "final_csa": float(csa_time),
            "total": float(total_time),
        },
    }

    with open(config.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="CS-MOGA-ACSA-RBF for Application Layer DDoS Detection"
    )

    parser.add_argument("--csv", required=True, help="Path to CSV dataset.")
    parser.add_argument("--label-column", required=True, help="Label column name.")

    label_group = parser.add_mutually_exclusive_group(required=False)
    label_group.add_argument(
        "--attack-label",
        default=None,
        help="Value in label column that should be mapped to attack=1.",
    )
    label_group.add_argument(
        "--normal-label",
        default=None,
        help="Value in label column that should be mapped to normal=0; all others become attack=1.",
    )

    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)

    parser.add_argument("--hidden-units", type=int, default=12)

    parser.add_argument("--ga-population", type=int, default=24)
    parser.add_argument("--ga-generations", type=int, default=40)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.08)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.75)

    parser.add_argument("--csa-population", type=int, default=40)
    parser.add_argument("--csa-iterations", type=int, default=200)

    parser.add_argument("--final-csa-population", type=int, default=50)
    parser.add_argument("--final-csa-iterations", type=int, default=300)

    parser.add_argument(
        "--selected-features-json",
        default=None,
        help=(
            "Path to a JSON file containing a selected_features list. "
            "When provided, GA feature selection is skipped."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    return ExperimentConfig(
        csv=args.csv,
        label_column=args.label_column,
        attack_label=args.attack_label,
        normal_label=args.normal_label,
        test_size=args.test_size,
        validation_size=args.validation_size,
        hidden_units=args.hidden_units,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
        ga_mutation_rate=args.ga_mutation_rate,
        ga_crossover_rate=args.ga_crossover_rate,
        csa_population=args.csa_population,
        csa_iterations=args.csa_iterations,
        final_csa_population=args.final_csa_population,
        final_csa_iterations=args.final_csa_iterations,
        selected_features_json=args.selected_features_json,
        seed=args.seed,
        output=args.output,
        verbose=args.verbose,
    )


def main() -> None:
    config = parse_args()
    result = run_experiment(config)

    test = result["metrics"]["test"]

    print(
        json.dumps(
            {
                "method": result["method"],
                "selected_features_count": result["data"]["selected_features_count"],
                "encoded_features": result["data"]["encoded_features"],
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_precision": test["precision"],
                "test_recall_detection_rate": test["recall_detection_rate"],
                "test_f1": test["f1"],
                "test_mae": test["mae"],
                "test_fpr": test["fpr"],
                "test_fnr": test["fnr"],
                "test_roc_auc": test["roc_auc"],
                "test_pr_auc": test["pr_auc"],
                "output": config.output,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
