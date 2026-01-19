from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sklearn.feature_selection import f_classif, f_regression

from .models import CLASSIFICATION_PIPELINES, REGRESSION_PIPELINES

DEFAULT_FACTOR_MULTIPLIERS: Tuple[float, ...] = (0.5, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3)
DEFAULT_RULE_GENERATORS: Dict[str, bool] = {
    "Reg_CART": False,
    "Clas_CART": True,
    "ORT": False,
    "OCT": True,
    "ORT_H": False,
    "OCT_H": True,
}
DEFAULT_BASELINES: Dict[str, Tuple[str, ...]] = {
    "classification": ("logistic_regression", "svm", "naive_bayes", "knn"),
    "regression": ("linear_regression", "svm", "knn"),
}
DEFAULT_RULE_MODELS: Dict[str, Tuple[str, ...]] = {
    "classification": ("logistic_regression", "svm", "naive_bayes", "knn"),
    "regression": ("linear_regression", "svm", "knn"),
}
METRIC_BY_PROBLEM: Dict[str, str] = {"classification": "accuracy", "regression": "r2"}
SELECTOR_BY_PROBLEM: Dict[str, Any] = {"classification": f_classif, "regression": f_regression}
RULE_PERFORMANCE_INDICES: Dict[str, List[Tuple[int, str]]] = {
    "classification": [(1, "CART"), (3, "OCT"), (5, "OCT-H")],
    "regression": [(0, "Reg-CART"), (2, "ORT"), (4, "ORT-H")],
}
RULE_MODEL_INDICES: Dict[str, List[Tuple[int, str]]] = {
    "classification": [(1, "CART"), (3, "OCT"), (5, "OCT-H"), (7, "OCT+OCT-H")],
    "regression": [(0, "Reg-CART"), (2, "ORT"), (4, "ORT-H"), (6, "ORT+ORT-H")],
}


@dataclass(frozen=True)
class RuleConfig:
    enabled: bool = True
    feature_subsample_size: Optional[int] = None
    n_num: int = 1
    max_iter_hy: int = 2
    depth_grid: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    depth_grid_hy: Tuple[int, ...] = (1, 2)
    complexity_bi: float = 0.001
    complexity_hy: float = 0.001
    generators: Optional[Dict[str, bool]] = None

    def __post_init__(self) -> None:
        if self.generators is None:
            object.__setattr__(self, "generators", dict(DEFAULT_RULE_GENERATORS))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RuleConfig":
        raw = raw or {}
        gens = raw.get("generators", {}) or {}
        merged_generators = {k: bool(gens.get(k, v)) for k, v in DEFAULT_RULE_GENERATORS.items()}
        feature_subsample_size = raw.get("feature_subsample_size")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            feature_subsample_size=int(feature_subsample_size) if feature_subsample_size is not None else None,
            n_num=int(raw.get("n_num", 1)),
            max_iter_hy=int(raw.get("max_iter_hy", 2)),
            depth_grid=tuple(int(x) for x in raw.get("depth_grid", (1, 2, 3, 4, 5, 6))),
            depth_grid_hy=tuple(int(x) for x in raw.get("depth_grid_hy", (1, 2))),
            complexity_bi=float(raw.get("complexity_bi", 0.001)),
            complexity_hy=float(raw.get("complexity_hy", 0.001)),
            generators=merged_generators,
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    problem_type: str
    dataset_names: Optional[List[str]]
    dataset_start: int
    dataset_end: int
    iters: int
    test_size: float
    max_rows: int
    max_cols: int
    max_dummy_ratio: float
    require_cols_lt_rows: bool
    factor_multipliers: Tuple[float, ...]
    baseline_models: Tuple[str, ...]
    rule_models: Tuple[str, ...]
    rules: RuleConfig
    n_jobs: int
    outdir: str
    checkpoint_dir: Optional[str]
    loglevel: str

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        problem_type = str(raw.get("problem_type", "classification")).lower()
        if problem_type not in ("classification", "regression"):
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        baseline_models = raw.get("baseline_models")
        rule_models = raw.get("rule_models")
        baseline_models = tuple(baseline_models) if baseline_models is not None else DEFAULT_BASELINES[problem_type]
        rule_models = tuple(rule_models) if rule_models is not None else DEFAULT_RULE_MODELS[problem_type]

        valid_registry = CLASSIFICATION_PIPELINES if problem_type == "classification" else REGRESSION_PIPELINES
        invalid_baselines = [m for m in baseline_models if m not in valid_registry]
        invalid_rules = [m for m in rule_models if m not in valid_registry]
        if invalid_baselines or invalid_rules:
            raise ValueError(
                f"Unknown model(s) for {problem_type}: "
                f"baselines={invalid_baselines} rules={invalid_rules}. "
                f"Valid options: {sorted(valid_registry.keys())}"
            )

        factor_multipliers = tuple(float(x) for x in raw.get("factor_multipliers", DEFAULT_FACTOR_MULTIPLIERS))
        rules_cfg = RuleConfig.from_dict(raw.get("rules", {}))

        checkpoint_dir = raw.get("checkpoint_dir")
        checkpoint_dir = str(checkpoint_dir) if checkpoint_dir else None

        dataset_names = raw.get("dataset_names")
        dataset_names = list(dataset_names) if dataset_names else None

        return cls(
            problem_type=problem_type,
            dataset_names=dataset_names,
            dataset_start=int(raw.get("dataset_start", 0)),
            dataset_end=int(raw.get("dataset_end", 10)),
            iters=int(raw.get("iters", 5)),
            test_size=float(raw.get("test_size", 0.2)),
            max_rows=int(raw.get("max_rows", 50_000)),
            max_cols=int(raw.get("max_cols", 100)),
            max_dummy_ratio=float(raw.get("max_dummy_ratio", 0.30)),
            require_cols_lt_rows=bool(raw.get("require_cols_lt_rows", True)),
            factor_multipliers=factor_multipliers,
            baseline_models=baseline_models,
            rule_models=rule_models,
            rules=rules_cfg,
            n_jobs=int(raw.get("n_jobs", 10)),
            outdir=str(raw.get("outdir", "outputs")),
            checkpoint_dir=checkpoint_dir,
            loglevel=str(raw.get("loglevel", "INFO")),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rules"] = asdict(self.rules)
        return data
