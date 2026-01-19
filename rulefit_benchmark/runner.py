from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pmlb import classification_dataset_names as pmlb_classification_dataset_names
from pmlb import regression_dataset_names as pmlb_regression_dataset_names
from pmlb import fetch_data
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .config import (
    BenchmarkConfig,
    METRIC_BY_PROBLEM,
    RULE_MODEL_INDICES,
    RULE_PERFORMANCE_INDICES,
    SELECTOR_BY_PROBLEM,
)
from .models import get_pipeline
from .preprocessing import dataset_is_too_big, make_factor_grid, one_hot_if_reasonable

LOGGER = logging.getLogger("benchmark")


class TqdmParallel(Parallel):
    """joblib.Parallel with a tqdm progress bar."""

    def __init__(self, total: Optional[int] = None, *args: Any, **kwargs: Any):
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        with tqdm(total=self._total, desc="Datasets", unit="ds") as self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def _active_rule_flags(cfg: BenchmarkConfig) -> Dict[str, bool]:
    if cfg.problem_type == "classification":
        keys = ("Clas_CART", "OCT", "OCT_H")
    else:
        keys = ("Reg_CART", "ORT", "ORT_H")
    return {k: bool(cfg.rules.generators.get(k, False)) for k in keys}


def _should_generate_rules(cfg: BenchmarkConfig) -> bool:
    return cfg.rules.enabled and any(_active_rule_flags(cfg).values())


def _load_rule_tools():
    try:
        from . import rules  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Rule generation requested but rulefit_benchmark.rules (and interpretableai) is not available."
        ) from exc
    return rules.generate_tree, rules.gen_train_and_test_features


def run_single_dataset(
    dataset_name: str,
    cfg: BenchmarkConfig,
    checkpoint_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full benchmark for a single PMLB dataset."""
    target_col = "target"
    rows: List[Dict[str, Any]] = []
    metric_name = METRIC_BY_PROBLEM[cfg.problem_type]
    selector_fn = SELECTOR_BY_PROBLEM[cfg.problem_type]
    rules_active = _should_generate_rules(cfg)
    rule_tools: Optional[Tuple[Any, Any]] = None

    try:
        try:
            df = fetch_data(dataset_name)
        except ValueError as e:
            LOGGER.info("Skipping %s (%s)", dataset_name, e)
            return pd.DataFrame(rows)

        if dataset_is_too_big(df, cfg):
            LOGGER.info("Skipping %s (too big: %s)", dataset_name, df.shape)
            return pd.DataFrame(rows)

        df, cont_cols, bin_cols, cat_cols = one_hot_if_reasonable(df, target_col, cfg)
        if df.empty:
            LOGGER.info("Skipping %s (one-hot expansion too large)", dataset_name)
            return pd.DataFrame(rows)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        LOGGER.info(
            "Dataset=%s rows=%d cols=%d target_unique=%d cont=%d bin=%d cat=%d",
            dataset_name,
            X.shape[0],
            X.shape[1],
            y.nunique(),
            len(cont_cols),
            len(bin_cols),
            len(cat_cols),
        )

        factors = make_factor_grid(len(X.columns), cfg.factor_multipliers)
        scaler = StandardScaler()
        split_kwargs: Dict[str, Any] = {
            "test_size": cfg.test_size,
            "random_state": None,
        }
        if cfg.problem_type == "classification":
            split_kwargs["stratify"] = y

        for it in range(cfg.iters):
            split_kwargs["random_state"] = it
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                **split_kwargs,
            )

            X_cols = X_train.columns
            X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_cols)
            X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_cols)

            if rules_active:
                if rule_tools is None:
                    rule_tools = _load_rule_tools()
                generate_tree_fn, gen_train_and_test_features_fn = rule_tools

                feat_size = cfg.rules.feature_subsample_size
                feat_size = len(X.columns) if feat_size is None else int(feat_size)

                generator_flags = _active_rule_flags(cfg)
                models, performance = generate_tree_fn(
                    X_train_s,
                    y_train,
                    X_test_s,
                    y_test,
                    n_num=int(cfg.rules.n_num),
                    feat_size=feat_size,
                    max_iter_hy=cfg.rules.max_iter_hy,
                    depth_grid=cfg.rules.depth_grid,
                    depth_grid_hy=cfg.rules.depth_grid_hy,
                    complexity_bi=cfg.rules.complexity_bi,
                    complexity_hy=cfg.rules.complexity_hy,
                    Reg_CART=generator_flags.get("Reg_CART", False),
                    ORT=generator_flags.get("ORT", False),
                    ORT_H=generator_flags.get("ORT_H", False),
                    Clas_CART=generator_flags.get("Clas_CART", False),
                    OCT=generator_flags.get("OCT", False),
                    OCT_H=generator_flags.get("OCT_H", False),
                )

                for idx, name in RULE_PERFORMANCE_INDICES[cfg.problem_type]:
                    perf = performance[idx]
                    if perf and not (isinstance(perf, float) and np.isnan(perf)):
                        perf_score = np.nanmean(perf)
                        if np.isnan(perf_score):
                            continue
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "model": name,
                                "iter": it,
                                "feature_factor": 1.0,
                                "metric": metric_name,
                                "score": float(perf_score),
                            }
                        )

                active_names: List[str] = []
                active_rules: List[Any] = []
                for idx, name in RULE_MODEL_INDICES[cfg.problem_type]:
                    model_obj = models[idx]
                    if model_obj:
                        active_names.append(name)
                        active_rules.append(model_obj)

                if active_rules:
                    datasets = gen_train_and_test_features_fn(active_rules, active_names, X_train_s, X_test_s)

                    for model_name, pair in datasets.items():
                        (X_train_rules_and_features, X_test_rules_and_features) = pair[0]
                        (X_train_only_rules, X_test_only_rules) = pair[1]

                        n_rules = X_train_only_rules.shape[1]
                        if n_rules == 0:
                            continue

                        for k, fac_name in factors:
                            k_rules = min(k, n_rules)
                            if k_rules <= 0:
                                continue
                            k_rules_fac_name = (k / len(X.columns)) if k < n_rules else 1.0

                            if (
                                k <= X_train_s.shape[0]
                                and k <= X_train_rules_and_features.shape[1]
                                and k_rules <= X_train_only_rules.shape[1]
                            ):
                                selector_rf = SelectKBest(score_func=selector_fn, k=k).fit(
                                    X_train_rules_and_features, y_train
                                )
                                cols_rf = selector_rf.get_feature_names_out()
                                Xtr_rf = X_train_rules_and_features[cols_rf]
                                Xte_rf = X_test_rules_and_features[cols_rf]

                                selector_r = SelectKBest(score_func=selector_fn, k=k_rules).fit(
                                    X_train_only_rules, y_train
                                )
                                cols_r = selector_r.get_feature_names_out()
                                Xtr_r = X_train_only_rules[cols_r]
                                Xte_r = X_test_only_rules[cols_r]

                                for model_key in cfg.rule_models:
                                    pipeline = get_pipeline(cfg.problem_type, model_key)
                                    rows.append(
                                        {
                                            "dataset": dataset_name,
                                            "model": f"{model_name}_{model_key}_rules",
                                            "iter": it,
                                            "feature_factor": float(k_rules_fac_name),
                                            "metric": metric_name,
                                            "score": float(pipeline(Xtr_r, Xte_r, y_train, y_test)),
                                        }
                                    )
                                    rows.append(
                                        {
                                            "dataset": dataset_name,
                                            "model": f"{model_name}_{model_key}_rules_and_features",
                                            "iter": it,
                                            "feature_factor": float(fac_name),
                                            "metric": metric_name,
                                            "score": float(pipeline(Xtr_rf, Xte_rf, y_train, y_test)),
                                        }
                                    )

            for model_key in cfg.baseline_models:
                pipeline = get_pipeline(cfg.problem_type, model_key)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_key,
                        "iter": it,
                        "feature_factor": 1.0,
                        "metric": metric_name,
                        "score": float(pipeline(X_train_s, X_test_s, y_train, y_test)),
                    }
                )

            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoint_dir / f"{dataset_name}_partial.pkl"
                with ckpt_path.open("wb") as f:
                    pickle.dump(rows, f)

        return pd.DataFrame(rows)

    except Exception:
        LOGGER.exception("Failed dataset=%s", dataset_name)
        return pd.DataFrame(rows)


def run_benchmark(
    datasets: List[str],
    cfg: BenchmarkConfig,
    checkpoint_dir: Optional[Path] = None,
) -> pd.DataFrame:
    LOGGER.info("Running %d datasets with config: %s", len(datasets), cfg.to_dict())
    results = TqdmParallel(total=len(datasets), n_jobs=cfg.n_jobs)(
        delayed(run_single_dataset)(ds, cfg, checkpoint_dir) for ds in datasets
    )
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True) if len(results) > 1 else results[0]


def available_datasets(problem_type: str) -> List[str]:
    return list(pmlb_classification_dataset_names) if problem_type == "classification" else list(pmlb_regression_dataset_names)

