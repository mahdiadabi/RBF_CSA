from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .metrics import classification_metrics, regression_metrics
from .nsl_kdd import (
    FEATURES,
    build_preprocessor,
    original_feature_groups,
    read_binary_nsl_kdd,
)

try:
    import torch
except ImportError:
    torch = None


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


class RBFClassifier:
    def __init__(
        self,
        input_size: int,
        hidden_units: int,
    ):
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.solution_size = hidden_units * (input_size + 2) + 1

    def unpack(
        self,
        solution: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]:
        center_end = self.hidden_units * self.input_size

        centers = solution[:center_end].reshape(
            self.hidden_units,
            self.input_size,
        )

        spreads = np.maximum(
            solution[center_end : center_end + self.hidden_units],
            1e-4,
        )

        weights = solution[
            center_end + self.hidden_units : center_end + 2 * self.hidden_units
        ]

        bias = float(solution[-1])

        return centers, spreads, weights, bias

    def scores(
        self,
        solution: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        centers, spreads, weights, bias = self.unpack(solution)

        squared = np.sum(
            (inputs[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )

        basis = np.exp(-squared / (2.0 * spreads[None, :] ** 2))

        return sigmoid(basis @ weights + bias)


@dataclass
class CSAConfig:
    population_size: int = 50
    iterations: int = 500
    discovery_probability: float = 0.25
    alpha: float = 1.0
    levy_beta: float = 1.5
    epsilon: float = 1e-6
    seed: int = 42
    device: str | None = None
    loss_batch_size: int = 8


class CuckooSearch:
    def __init__(
        self,
        model: RBFClassifier,
        config: CSAConfig | None = None,
    ):
        self.model = model
        self.config = config or CSAConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.best_solution_: np.ndarray | None = None
        self.best_mse_: float = np.inf
        self.device_ = self._resolve_device()

    def _resolve_device(self):
        if torch is None:
            if self.config.device is not None:
                raise RuntimeError(
                    "A PyTorch device was requested, but PyTorch is not installed. "
                    "Install the project GPU extra or use automatic device selection."
                )
            return None

        if self.config.device is not None:
            device = torch.device(self.config.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-enabled PyTorch device is available. "
                "Use device='cpu' or install a CUDA-enabled PyTorch build."
            )

        return device

    def _levy(
        self,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        beta = self.config.levy_beta

        sigma = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        numerator = self.rng.normal(
            0,
            sigma,
            shape,
        )

        denominator = np.abs(
            self.rng.normal(
                0,
                1,
                shape,
            )
        ) ** (1 / beta)

        return numerator / np.maximum(
            denominator,
            1e-12,
        )

    def _losses(
        self,
        nests: np.ndarray,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        batch_size = max(1, self.config.loss_batch_size)
        center_count = self.model.hidden_units * self.model.input_size
        input_norm = np.sum(inputs**2, axis=1)
        losses: list[np.ndarray] = []

        for start in range(0, len(nests), batch_size):
            batch = nests[start : start + batch_size]
            centers = batch[:, :center_count].reshape(
                len(batch),
                self.model.hidden_units,
                self.model.input_size,
            )
            spreads = np.maximum(
                batch[:, center_count : center_count + self.model.hidden_units],
                1e-4,
            )
            weights = batch[
                :,
                center_count
                + self.model.hidden_units : center_count
                + 2 * self.model.hidden_units,
            ]
            bias = batch[:, -1]

            squared = (
                input_norm[None, :, None]
                + np.sum(centers**2, axis=2)[:, None, :]
                - 2.0 * np.einsum("nd,bhd->bnh", inputs, centers)
            )
            basis = np.exp(
                -np.maximum(squared, 0.0)
                / (2.0 * spreads[:, None, :] ** 2)
            )
            scores = sigmoid(
                np.sum(basis * weights[:, None, :], axis=2) + bias[:, None]
            )
            losses.append(np.mean((scores - targets[None, :]) ** 2, axis=1))

        return np.concatenate(losses)

    def _torch_losses(
        self,
        nests,
        inputs,
        targets,
        input_norm,
    ):
        batch_size = max(1, self.config.loss_batch_size)
        center_count = self.model.hidden_units * self.model.input_size
        losses = []

        for start in range(0, len(nests), batch_size):
            batch = nests[start : start + batch_size]
            centers = batch[:, :center_count].reshape(
                len(batch),
                self.model.hidden_units,
                self.model.input_size,
            )
            spreads = torch.clamp(
                batch[:, center_count : center_count + self.model.hidden_units],
                min=1e-4,
            )
            weights = batch[
                :,
                center_count
                + self.model.hidden_units : center_count
                + 2 * self.model.hidden_units,
            ]
            bias = batch[:, -1]

            squared = (
                input_norm[None, :, None]
                + centers.square().sum(dim=2)[:, None, :]
                - 2.0 * torch.einsum("nd,bhd->bnh", inputs, centers)
            )
            basis = torch.exp(
                -torch.clamp(squared, min=0.0)
                / (2.0 * spreads[:, None, :].square())
            )
            scores = torch.sigmoid(
                (basis * weights[:, None, :]).sum(dim=2) + bias[:, None]
            )
            losses.append(((scores - targets[None, :]) ** 2).mean(dim=1))

        return torch.cat(losses)

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        progress_desc: str | None = None,
    ) -> "CuckooSearch":
        if self.device_ is None:
            return self._fit_numpy(inputs, targets, progress_desc)

        return self._fit_torch(inputs, targets, progress_desc)

    def _fit_numpy(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        progress_desc: str | None,
    ) -> "CuckooSearch":
        dimensions = self.model.solution_size

        nests = self.rng.uniform(
            -1,
            1,
            (
                self.config.population_size,
                dimensions,
            ),
        )

        center_count = self.model.hidden_units * self.model.input_size

        spread_slice = slice(
            center_count,
            center_count + self.model.hidden_units,
        )

        nests[:, spread_slice] = self.rng.uniform(
            0.05,
            1.0,
            (
                len(nests),
                self.model.hidden_units,
            ),
        )

        losses = self._losses(
            nests,
            inputs,
            targets,
        )

        progress = tqdm(
            range(self.config.iterations),
            desc=progress_desc,
            unit="iteration",
            disable=progress_desc is None,
            dynamic_ncols=True,
        )

        for _ in progress:
            best_index = int(np.argmin(losses))

            best = nests[best_index].copy()

            if losses[best_index] < self.best_mse_:
                self.best_mse_ = float(losses[best_index])

                self.best_solution_ = best.copy()

            if progress_desc is not None:
                progress.set_postfix(
                    best_mse=(f"{self.best_mse_:.6f}"),
                    refresh=False,
                )

            if self.best_mse_ <= self.config.epsilon:
                break

            step = self.config.alpha * self._levy(nests.shape) * (nests - best)

            candidates = nests + 0.01 * step

            candidates[:, spread_slice] = np.clip(
                candidates[:, spread_slice],
                1e-4,
                2.0,
            )

            candidate_losses = self._losses(
                candidates,
                inputs,
                targets,
            )

            improved = candidate_losses < losses

            nests[improved] = candidates[improved]

            losses[improved] = candidate_losses[improved]

            first, second = self.rng.permutation(len(nests))[:2]

            mask = self.rng.random(nests.shape) < self.config.discovery_probability

            local = nests + mask * self.rng.random(nests.shape) * (
                nests[first] - nests[second]
            )

            local[:, spread_slice] = np.clip(
                local[:, spread_slice],
                1e-4,
                2.0,
            )

            local_losses = self._losses(
                local,
                inputs,
                targets,
            )

            improved = local_losses < losses

            nests[improved] = local[improved]
            losses[improved] = local_losses[improved]

        index = int(np.argmin(losses))

        if losses[index] < self.best_mse_:
            self.best_mse_ = float(losses[index])

            self.best_solution_ = nests[index].copy()

        return self

    def _fit_torch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        progress_desc: str | None,
    ) -> "CuckooSearch":
        device = self.device_
        dimensions = self.model.solution_size
        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.seed)
        inputs_tensor = torch.as_tensor(
            np.asarray(inputs, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        targets_tensor = torch.as_tensor(
            np.asarray(targets, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        input_norm = inputs_tensor.square().sum(dim=1)

        nests = torch.empty(
            (self.config.population_size, dimensions),
            device=device,
        ).uniform_(-1, 1, generator=generator)

        center_count = self.model.hidden_units * self.model.input_size
        spread_slice = slice(
            center_count,
            center_count + self.model.hidden_units,
        )

        nests[:, spread_slice] = torch.empty(
            (len(nests), self.model.hidden_units),
            device=device,
        ).uniform_(0.05, 1.0, generator=generator)

        losses = self._torch_losses(
            nests,
            inputs_tensor,
            targets_tensor,
            input_norm,
        )
        beta = self.config.levy_beta
        sigma = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        progress = tqdm(
            range(self.config.iterations),
            desc=progress_desc,
            unit="iteration",
            disable=progress_desc is None,
            dynamic_ncols=True,
        )

        for _ in progress:
            best_index = int(torch.argmin(losses).item())
            best = nests[best_index].clone()
            best_loss = float(losses[best_index].item())

            if best_loss < self.best_mse_:
                self.best_mse_ = best_loss
                self.best_solution_ = best.detach().cpu().numpy().copy()

            if progress_desc is not None:
                progress.set_postfix(
                    best_mse=f"{self.best_mse_:.6f}",
                    refresh=False,
                )

            if self.best_mse_ <= self.config.epsilon:
                break

            levy = (
                torch.randn(nests.shape, device=device, generator=generator) * sigma
                / torch.clamp(
                    torch.abs(
                        torch.randn(
                            nests.shape,
                            device=device,
                            generator=generator,
                        )
                    )
                    ** (1 / beta),
                    min=1e-12,
                )
            )
            candidates = nests + 0.01 * self.config.alpha * levy * (nests - best)
            candidates[:, spread_slice].clamp_(1e-4, 2.0)

            candidate_losses = self._torch_losses(
                candidates,
                inputs_tensor,
                targets_tensor,
                input_norm,
            )
            improved = candidate_losses < losses
            nests[improved] = candidates[improved]
            losses[improved] = candidate_losses[improved]

            first, second = torch.randperm(
                len(nests),
                device=device,
                generator=generator,
            )[:2]
            mask = (
                torch.rand(
                    nests.shape,
                    device=device,
                    generator=generator,
                )
                < self.config.discovery_probability
            )
            local = nests + mask * torch.rand(
                nests.shape,
                device=device,
                generator=generator,
            ) * (nests[first] - nests[second])
            local[:, spread_slice].clamp_(1e-4, 2.0)

            local_losses = self._torch_losses(
                local,
                inputs_tensor,
                targets_tensor,
                input_norm,
            )
            improved = local_losses < losses
            nests[improved] = local[improved]
            losses[improved] = local_losses[improved]

        index = int(torch.argmin(losses).item())

        if float(losses[index].item()) < self.best_mse_:
            self.best_mse_ = float(losses[index].item())
            self.best_solution_ = nests[index].detach().cpu().numpy().copy()

        return self


@dataclass
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

    def _repair(
        self,
        population: np.ndarray,
    ) -> np.ndarray:
        for row in population:
            active = np.flatnonzero(row)
            inactive = np.flatnonzero(~row)

            if len(active) > self.config.selected_count:
                selected = self.rng.choice(
                    active,
                    (len(active) - self.config.selected_count),
                    replace=False,
                )

                row[selected] = False

            elif len(active) < self.config.selected_count:
                selected = self.rng.choice(
                    inactive,
                    (self.config.selected_count - len(active)),
                    replace=False,
                )

                row[selected] = True

        return population

    def _columns(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([self.groups[index] for index in np.flatnonzero(mask)])

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> "GroupedFeatureGA":
        population = self._repair(
            self.rng.random(
                (
                    self.config.population_size,
                    len(self.groups),
                )
            )
            < 0.25
        )

        (
            train_x,
            valid_x,
            train_y,
            valid_y,
        ) = train_test_split(
            inputs,
            targets,
            test_size=0.2,
            stratify=targets,
            random_state=self.config.seed,
        )

        cache: dict[bytes, float] = {}

        def loss(mask: np.ndarray) -> float:
            key = mask.tobytes()

            if key not in cache:
                columns = self._columns(mask)

                model = RBFClassifier(
                    len(columns),
                    self.hidden_units,
                )

                search = CuckooSearch(
                    model,
                    CSAConfig(
                        population_size=12,
                        iterations=25,
                        seed=(self.config.seed + len(cache)),
                    ),
                )

                search.fit(
                    train_x[:, columns],
                    train_y,
                )

                if search.best_solution_ is None:
                    raise RuntimeError("Internal CSA did not " "produce a solution")

                scores = model.scores(
                    search.best_solution_,
                    valid_x[:, columns],
                )
                cache[key] = float(np.mean((valid_y - scores) ** 2))

            return cache[key]

        best = population[0].copy()
        best_loss = np.inf

        total_evaluations = self.config.iterations * self.config.population_size

        progress = tqdm(
            total=total_evaluations,
            desc="Feature GA",
            unit="candidate",
            dynamic_ncols=True,
        )

        try:
            for generation in range(self.config.iterations):
                generation_losses: list[float] = []

                for candidate_index, mask in enumerate(population):
                    key = mask.tobytes()
                    is_cached = key in cache

                    progress.set_postfix(
                        generation=(f"{generation + 1}/" f"{self.config.iterations}"),
                        candidate=(
                            f"{candidate_index + 1}/" f"{self.config.population_size}"
                        ),
                        unique=len(cache),
                        cached=("yes" if is_cached else "no"),
                        best=(
                            "N/A" if not np.isfinite(best_loss) else f"{best_loss:.6f}"
                        ),
                        refresh=False,
                    )

                    generation_losses.append(loss(mask))

                    progress.update(1)

                losses = np.asarray(generation_losses)

                best_index = int(np.argmin(losses))

                if losses[best_index] < best_loss:
                    best = population[best_index].copy()

                    best_loss = float(losses[best_index])

                progress.set_postfix(
                    generation=(f"{generation + 1}/" f"{self.config.iterations}"),
                    unique=len(cache),
                    best=f"{best_loss:.6f}",
                    refresh=True,
                )

                candidates = self.rng.integers(
                    0,
                    len(population),
                    (
                        len(population),
                        self.config.tournament_size,
                    ),
                )

                candidate_losses = losses[candidates]

                tournament_winners = np.argmin(
                    candidate_losses,
                    axis=1,
                )

                winners = candidates[
                    np.arange(len(population)),
                    tournament_winners,
                ]

                children = population[winners].copy()

                for child_index in range(
                    0,
                    len(children) - 1,
                    2,
                ):
                    if self.rng.random() < self.config.crossover_probability:
                        point = self.rng.integers(
                            1,
                            len(self.groups),
                        )

                        tail = children[
                            child_index,
                            point:,
                        ].copy()

                        children[
                            child_index,
                            point:,
                        ] = children[
                            child_index + 1,
                            point:,
                        ]

                        children[
                            child_index + 1,
                            point:,
                        ] = tail

                mutation_rows = (
                    self.rng.random(len(children)) < self.config.mutation_probability
                )

                for row in np.flatnonzero(mutation_rows):
                    bit = self.rng.integers(
                        0,
                        len(self.groups),
                    )

                    children[row, bit] = ~children[row, bit]

                population = self._repair(children)

                population[0] = best

        finally:
            progress.close()

        self.best_mask_ = best
        self.best_loss_ = best_loss

        return self

    def selected_columns(self) -> np.ndarray:
        if self.best_mask_ is None:
            raise RuntimeError("Feature GA is not fitted")

        return self._columns(self.best_mask_)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "NSL-KDD feature selection using GA " "and RBF training using Cuckoo Search"
        )
    )

    parser.add_argument(
        "nsl_kdd",
        help="Path to KDDTrain+.txt",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--hidden-units",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--ga-iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--csa-iterations",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device for CSA optimization (default: auto)",
    )

    args = parser.parse_args()

    print("Loading NSL-KDD dataset...", flush=True)

    frame, targets = read_binary_nsl_kdd(args.nsl_kdd)

    print(
        f"Loaded {len(frame):,} samples.",
        flush=True,
    )

    (
        train_frame,
        test_frame,
        train_y,
        test_y,
    ) = train_test_split(
        frame,
        targets,
        test_size=0.2,
        stratify=targets,
        random_state=args.seed,
    )

    print(
        f"Training samples: {len(train_frame):,}",
        flush=True,
    )

    print(
        f"Testing samples: {len(test_frame):,}",
        flush=True,
    )

    print("Preprocessing features...", flush=True)

    preprocessor = build_preprocessor(train_frame)

    train_x = preprocessor.fit_transform(train_frame)

    test_x = preprocessor.transform(test_frame)

    groups = original_feature_groups(preprocessor)

    print(
        f"Encoded feature count: " f"{train_x.shape[1]}",
        flush=True,
    )

    print(
        "Starting GA feature selection...",
        flush=True,
    )

    feature_ga = GroupedFeatureGA(
        groups,
        args.hidden_units,
        FeatureGAConfig(
            iterations=args.ga_iterations,
            seed=args.seed,
        ),
    ).fit(
        train_x,
        train_y,
    )

    columns = feature_ga.selected_columns()

    selected_features = [
        name
        for name, chosen in zip(
            FEATURES,
            feature_ga.best_mask_,
        )
        if chosen
    ]

    print(
        "Selected original features:",
        flush=True,
    )

    for feature_name in selected_features:
        print(
            f"  - {feature_name}",
            flush=True,
        )

    print(
        f"Selected encoded columns: " f"{len(columns)}",
        flush=True,
    )

    print(
        "Starting final CSA training...",
        flush=True,
    )

    model = RBFClassifier(
        len(columns),
        args.hidden_units,
    )

    search = CuckooSearch(
        model,
        CSAConfig(
            iterations=args.csa_iterations,
            seed=args.seed,
            device=None if args.device == "auto" else args.device,
        ),
    ).fit(
        train_x[:, columns],
        train_y,
        progress_desc="Final CSA",
    )

    if search.best_solution_ is None:
        raise RuntimeError("Final CSA did not produce a solution")

    print(
        "Evaluating test set...",
        flush=True,
    )

    scores = model.scores(
        search.best_solution_,
        test_x[:, columns],
    )

    predictions = (scores >= 0.5).astype(int)

    result = regression_metrics(
        test_y,
        scores,
    )

    result.update(
        classification_metrics(
            test_y,
            predictions,
        )
    )

    result["feature_ga_best_mse"] = feature_ga.best_loss_

    result["final_csa_best_mse"] = search.best_mse_

    result["selected_original_features"] = selected_features

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
