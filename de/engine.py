"""
Differential Evolution engine. Pure NumPy, no GPU dependencies.

All population operations (mutation, crossover, selection, scaling) run as
vectorized NumPy array ops. No Python loops over population members.

Port of aivalanche DE: DE/rand/3 + best-guided mutation, binomial crossover,
adaptive boundaries.
"""

import numpy as np
import os
import csv
import time
from typing import Callable, Optional, Dict, Any, List


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def _lhs(n_samples: int, n_dims: int) -> np.ndarray:
    """Latin Hypercube Sampling. Returns (n_samples, n_dims) in [0, 1]."""
    result = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = np.random.permutation(n_samples)
        result[:, j] = (perm + np.random.rand(n_samples)) / n_samples
    return result


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

TRANSFORM_NONE = 0
TRANSFORM_LOG = 1
TRANSFORM_NEGLOG = 2


def load_parameters(path: str) -> Dict[str, Any]:
    """Load parameters from CSV. Returns dict with arrays for DE.

    CSV columns: name, min, max, scale
    Returns: {names, n_params, bounds_min, bounds_max, bounds_range, transforms}
    """
    import pandas as pd
    df = pd.read_csv(path)

    for col in ["name", "min", "max"]:
        if col not in df.columns:
            raise ValueError(f"parameters.csv must have column: {col}")

    df = df.sort_values("name").reset_index(drop=True)
    df[["min", "max"]] = df[["min", "max"]].astype(np.float64)
    if "scale" not in df.columns:
        df["scale"] = "lin"

    names = df["name"].tolist()
    n = len(names)

    # Determine transforms
    transforms = np.zeros(n, dtype=np.int32)
    for i, row in df.iterrows():
        if row["scale"].strip().lower() == "log":
            if row["min"] > 0 and row["max"] > 0:
                transforms[i] = TRANSFORM_LOG
            elif row["min"] < 0 and row["max"] < 0:
                transforms[i] = TRANSFORM_NEGLOG

    # Scale bounds
    mins = df["min"].values.astype(np.float64)
    maxs = df["max"].values.astype(np.float64)
    mins_scaled = _scale_array(mins, transforms)
    maxs_scaled = _scale_array(maxs, transforms)

    bounds_min = np.minimum(mins_scaled, maxs_scaled)
    bounds_max = np.maximum(mins_scaled, maxs_scaled)

    return {
        "names": names,
        "n_params": n,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "bounds_range": bounds_max - bounds_min,
        "transforms": transforms,
    }


def _scale_array(values: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Scale real values to optimization space."""
    result = values.copy()
    log_mask = transforms == TRANSFORM_LOG
    neglog_mask = transforms == TRANSFORM_NEGLOG
    if log_mask.any():
        result[log_mask] = np.log10(values[log_mask])
    if neglog_mask.any():
        result[neglog_mask] = np.log10(-values[neglog_mask])
    return result


def _unscale_array(values: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Unscale from optimization space to real values. Works on 1D or 2D."""
    result = values.copy()
    log_mask = transforms == TRANSFORM_LOG
    neglog_mask = transforms == TRANSFORM_NEGLOG
    if log_mask.any():
        result[..., log_mask] = 10.0 ** values[..., log_mask]
    if neglog_mask.any():
        result[..., neglog_mask] = -(10.0 ** values[..., neglog_mask])
    return result


def _normalize(x_scaled: np.ndarray, bounds_min: np.ndarray,
               bounds_range: np.ndarray) -> np.ndarray:
    """Normalize scaled values to [0, 1]."""
    return (x_scaled - bounds_min) / bounds_range


def _unnormalize(x_normed: np.ndarray, bounds_min: np.ndarray,
                 bounds_range: np.ndarray) -> np.ndarray:
    """Unnormalize from [0, 1] to scaled space."""
    return bounds_min + x_normed * bounds_range


def _to_real(x_normed: np.ndarray, params: Dict) -> np.ndarray:
    """Full pipeline: normalized [0,1] -> real values."""
    scaled = _unnormalize(x_normed, params["bounds_min"], params["bounds_range"])
    return _unscale_array(scaled, params["transforms"])


def _to_dicts(x_normed: np.ndarray, params: Dict) -> List[Dict[str, float]]:
    """Convert normalized population to list of param dicts for eval func."""
    real = _to_real(x_normed, params)
    names = params["names"]
    return [{name: float(row[i]) for i, name in enumerate(names)} for row in real]


# ---------------------------------------------------------------------------
# Differential Evolution
# ---------------------------------------------------------------------------

class DifferentialEvolution:
    """Differential Evolution optimizer. Pure NumPy, adaptive budget.

    Adaptive stopping: DE runs until the population truly converges, not
    until a fixed iteration count. It monitors population diversity (mean
    std across normalized parameters) and adjusts patience accordingly:

    - During warmup (< min_iterations): never stop.
    - Low diversity + no improvement for `patience` iters: converged.
    - High diversity + no improvement for `patience * 3` iters: stagnated.
    - Metric threshold reached: target hit.
    - Safety cap (max_iterations): hard limit, flags as not converged.

    The result dict includes `converged: bool` so callers know whether
    to trust the result.

    Args:
        params: Parameter dict from load_parameters().
        eval_func: f(parameters: list[dict], **kwargs) -> {"metrics": list[float]}
        pop_size: Population size.
        opt_dir: 'min' or 'max'.
        min_iterations: Warmup — never stop before this.
        max_iterations: Safety cap.
        metric_threshold: Stop when metric reaches this.
        patience: Base no-improve patience (extended when diversity is high).
        diversity_threshold: Population diversity below this = collapsed.
        F1, F2, F3: Mutation scaling factors.
        CR: Crossover probability.
        adaptive_bounds: Enable adaptive boundary expansion.
        results_dir: Directory to save results.
        eval_func_args: Extra kwargs passed to eval_func.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        eval_func: Callable,
        pop_size: int = 100,
        opt_dir: str = "min",
        min_iterations: int = 30,
        max_iterations: int = 5000,
        metric_threshold: float = 0.0,
        patience: int = 50,
        diversity_threshold: float = 0.01,
        F1: float = 0.8,
        F2: float = 0.8,
        F3: float = 0.0,
        CR: float = 0.9,
        adaptive_bounds: bool = False,
        adaptive_check_period: int = 10,
        adaptive_edge_threshold: float = 0.05,
        adaptive_pop_quantile: float = 0.7,
        adaptive_extension: float = 0.1,
        results_dir: Optional[str] = None,
        eval_func_args: Optional[Dict[str, Any]] = None,
    ):
        self.params = params
        self.n_params = params["n_params"]
        self.eval_func = eval_func
        self.eval_func_args = eval_func_args or {}
        self.pop_size = pop_size
        self.opt_dir = opt_dir
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.metric_threshold = metric_threshold
        self.patience = patience
        self.diversity_threshold = diversity_threshold
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.CR = CR
        self.adaptive_bounds = adaptive_bounds
        self.adaptive_check_period = adaptive_check_period
        self.adaptive_edge_threshold = adaptive_edge_threshold
        self.adaptive_pop_quantile = adaptive_pop_quantile
        self.adaptive_extension = adaptive_extension
        self.results_dir = results_dir

        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # State
        self.iteration = 0
        self.no_improve_count = 0
        self.best_metric = None
        self.best_normed = None
        self.best_real = None
        self.diversity = 1.0
        self.converged = False
        self.stop_reason = ""

    def run(self) -> Dict[str, Any]:
        """Run the full optimization loop."""
        self._init_population()

        while True:
            t0 = time.time()

            # Convert trials to param dicts for evaluation
            trial_dicts = _to_dicts(self.trials_normed, self.params)

            # Evaluate
            extra = {
                "iteration": self.iteration,
                "best_metric": self.best_metric,
                "best_parameters": self._best_as_dict(),
                **self.eval_func_args,
            }
            response = self.eval_func(parameters=trial_dicts, **extra)
            metrics = response["metrics"] if isinstance(response, dict) else response

            self.trials_metric = np.array(metrics, dtype=np.float64)

            # DE operations
            self._select_survivors()
            improved = self._update_best()

            # Track population diversity
            self.diversity = float(self.survivors_normed.std(axis=0).mean())

            dt = time.time() - t0
            print(f"[DE] iter {self.iteration:>4d} | best: {self.best_metric:.6e} | "
                  f"no_improve: {self.no_improve_count:>3d} | "
                  f"div: {self.diversity:.4f} | dt: {dt:.3f}s")

            if improved and self.results_dir:
                self._save_best()

            if self._check_stop():
                break

            self._next_generation()

        self._show_result()
        return {
            "best_parameters": self._best_as_dict(),
            "best_metric": self.best_metric,
            "iterations": self.iteration,
            "stop_reason": self.stop_reason,
            "converged": self.converged,
            "diversity": self.diversity,
        }

    # ----- Initialization -----

    def _init_population(self):
        self.iteration = 1
        self.no_improve_count = 0
        self.donors_normed = _lhs(self.pop_size, self.n_params)
        self.targets_normed = np.zeros((self.pop_size, self.n_params))
        self.targets_metric = np.full(self.pop_size,
                                       np.inf if self.opt_dir == "min" else -np.inf)
        self.trials_normed = self.donors_normed.copy()

    # ----- Core DE ops (all vectorized, no Python loops) -----

    def _mutate(self) -> np.ndarray:
        dice = np.random.randint(self.pop_size, size=(self.pop_size, 3))
        r1 = self.targets_normed[dice[:, 0]]
        r2 = self.targets_normed[dice[:, 1]]
        r3 = self.targets_normed[dice[:, 2]]

        donors = (r1
                  + self.F1 * (r2 - r3)
                  + self.F2 * (self.best_normed[np.newaxis, :] - r1)
                  + self.F3 * (self.best_normed[np.newaxis, :] - self.targets_normed))

        violation = (donors > 1.0) | (donors < 0.0)
        n_viol = violation.sum()
        if n_viol > 0:
            donors[violation] = np.random.rand(n_viol)

        return donors

    def _crossover(self, donors_normed: np.ndarray) -> np.ndarray:
        mask = np.random.rand(self.pop_size, self.n_params) < self.CR
        forced = np.random.randint(self.n_params, size=self.pop_size)
        mask[np.arange(self.pop_size), forced] = True
        return np.where(mask, donors_normed, self.targets_normed)

    def _select_survivors(self):
        if self.iteration == 1:
            self.survivors_normed = self.trials_normed.copy()
            self.survivors_metric = self.trials_metric.copy()
        else:
            if self.opt_dir == "max":
                mask = self.trials_metric > self.targets_metric
            else:
                mask = self.trials_metric < self.targets_metric
            mask_2d = mask[:, np.newaxis]
            self.survivors_normed = np.where(mask_2d, self.trials_normed, self.targets_normed)
            self.survivors_metric = np.where(mask, self.trials_metric, self.targets_metric)

    def _update_best(self) -> bool:
        if self.opt_dir == "max":
            idx = self.survivors_metric.argmax()
        else:
            idx = self.survivors_metric.argmin()

        candidate = self.survivors_metric[idx]

        if self.best_metric is None:
            improved = True
        elif self.opt_dir == "max":
            improved = candidate > self.best_metric
        else:
            improved = candidate < self.best_metric

        if improved:
            self.best_metric = float(candidate)
            self.best_normed = self.survivors_normed[idx].copy()
            self.best_real = _to_real(self.best_normed[np.newaxis, :], self.params).squeeze()
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        return improved

    def _adaptive_boundary_update(self):
        if not self.adaptive_bounds:
            return
        if self.iteration % self.adaptive_check_period != 0:
            return

        q_min = np.quantile(self.survivors_normed, self.adaptive_pop_quantile, axis=0)
        q_max = np.quantile(self.survivors_normed, 1.0 - self.adaptive_pop_quantile, axis=0)
        min_edge = q_min < self.adaptive_edge_threshold
        max_edge = q_max > (1.0 - self.adaptive_edge_threshold)

        if min_edge.any() or max_edge.any():
            old_min = self.params["bounds_min"].copy()
            old_range = self.params["bounds_range"].copy()

            scaled = old_min + self.survivors_normed * old_range

            self.params["bounds_min"][min_edge] -= self.adaptive_extension * old_range[min_edge]
            self.params["bounds_max"][max_edge] += self.adaptive_extension * old_range[max_edge]
            self.params["bounds_range"] = self.params["bounds_max"] - self.params["bounds_min"]

            self.survivors_normed = _normalize(scaled, self.params["bounds_min"],
                                                self.params["bounds_range"])
            if self.best_normed is not None:
                best_scaled = old_min + self.best_normed * old_range
                self.best_normed = _normalize(best_scaled, self.params["bounds_min"],
                                               self.params["bounds_range"])

    def _next_generation(self):
        self.iteration += 1
        self.targets_normed = self.survivors_normed
        self.targets_metric = self.survivors_metric

        if self.adaptive_bounds:
            self._adaptive_boundary_update()

        donors = self._mutate()
        self.trials_normed = self._crossover(donors)

    # ----- Stopping (adaptive budget) -----

    def _check_stop(self) -> bool:
        # 1. Target reached
        if self.best_metric is not None:
            if self.opt_dir == "min" and self.best_metric <= self.metric_threshold:
                self.stop_reason = "metric_threshold"
                self.converged = True
                return True
            if self.opt_dir == "max" and self.best_metric >= self.metric_threshold:
                self.stop_reason = "metric_threshold"
                self.converged = True
                return True

        # 2. Safety cap
        if self.iteration >= self.max_iterations:
            self.stop_reason = "max_iterations"
            self.converged = False
            return True

        # 3. No convergence checks during warmup
        if self.iteration < self.min_iterations:
            return False

        # 4. Adaptive stopping based on diversity
        if self.diversity < self.diversity_threshold:
            # Population collapsed — base patience is enough
            if self.no_improve_count >= self.patience:
                self.stop_reason = (f"converged (div={self.diversity:.4f}, "
                                    f"no_improve={self.no_improve_count})")
                self.converged = True
                return True
        else:
            # Population still diverse — give 3x patience to explore
            if self.no_improve_count >= self.patience * 3:
                self.stop_reason = (f"stagnated (div={self.diversity:.4f}, "
                                    f"no_improve={self.no_improve_count})")
                self.converged = True
                return True

        return False

    # ----- Utilities -----

    def _best_as_dict(self) -> Optional[Dict[str, float]]:
        if self.best_real is None:
            return None
        return {name: float(self.best_real[i])
                for i, name in enumerate(self.params["names"])}

    def _show_result(self):
        print(f"\n{'='*60}")
        print(f"  Stop reason:  {self.stop_reason}")
        print(f"  Converged:    {self.converged}")
        print(f"  Iterations:   {self.iteration}")
        print(f"  Best metric:  {self.best_metric:.6e}")
        print(f"  Diversity:    {self.diversity:.4f}")
        best = self._best_as_dict()
        if best:
            print(f"  Best parameters:")
            for k, v in best.items():
                print(f"    {k}: {v:.6e}")
        print(f"{'='*60}\n")

    def _save_best(self):
        if not self.results_dir:
            return
        best = self._best_as_dict()
        if not best:
            return
        path = os.path.join(self.results_dir, "best_parameters.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "value"])
            for k, v in best.items():
                w.writerow([k, v])
