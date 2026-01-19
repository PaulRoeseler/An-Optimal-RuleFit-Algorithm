from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import yaml

from .config import BenchmarkConfig
from .runner import available_datasets, run_benchmark

LOGGER = logging.getLogger("benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PMLB datasets (classification or regression) via YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--loglevel",
        type=str,
        default=None,
        help="Optional loglevel override (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = BenchmarkConfig.from_yaml(Path(args.config))
    loglevel = args.loglevel or cfg.loglevel

    logging.basicConfig(
        level=getattr(logging, loglevel.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
    if checkpoint_dir is not None and not checkpoint_dir.is_absolute():
        checkpoint_dir = outdir / checkpoint_dir

    all_names = available_datasets(cfg.problem_type)
    datasets = cfg.dataset_names if cfg.dataset_names is not None else all_names[cfg.dataset_start : cfg.dataset_end]
    if not datasets:
        LOGGER.warning("No datasets selected.")
        return

    (outdir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=True),
        encoding="utf-8",
    )

    df = run_benchmark(datasets=datasets, cfg=cfg, checkpoint_dir=checkpoint_dir)

    if df.empty:
        LOGGER.warning("No results produced.")
        return

    csv_path = outdir / "results.csv"
    df.to_csv(csv_path, index=False)

    try:
        pq_path = outdir / "results.parquet"
        df.to_parquet(pq_path, index=False)
    except Exception:
        LOGGER.info("Parquet not written (install pyarrow if you want it).")

    pkl_path = outdir / "results.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(df, f)

    LOGGER.info("Wrote results to: %s", csv_path)


if __name__ == "__main__":
    main()

