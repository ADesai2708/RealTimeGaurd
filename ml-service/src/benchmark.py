"""
benchmark.py

Inference latency benchmarking for the RealTimeGuard
fraud detection pipeline.

Measures:
  - Mean latency (ms)
  - Median / p50 latency (ms)
  - p95 latency (ms)
  - p99 latency (ms)
  - Max latency (ms)
  - Throughput (predictions / second)

Results are saved to reports/benchmark_results.json.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REPORT_DIR
from src.utils import save_json, setup_logger

logger = setup_logger(__name__)


class InferenceBenchmark:
    """
    Measures single-row prediction latency for a trained model.

    Single-row latency is the critical metric for real-time fraud
    detection because each incoming transaction is scored independently.

    Parameters
    ----------
    model : LGBMClassifier
        A fitted model that exposes predict_proba().
    X_sample : pd.DataFrame
        A representative sample of feature rows to benchmark against.
        Each iteration draws one row at random.
    n_iterations : int
        Number of timed prediction rounds. Default: 1000.
    report_dir : Path, optional
        Directory for output JSON. Defaults to REPORT_DIR from config.
    """

    def __init__(
        self,
        model,
        X_sample: pd.DataFrame,
        n_iterations: int = 1000,
        report_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.X_sample = X_sample.reset_index(drop=True)
        self.n_iterations = n_iterations
        self.report_dir = Path(report_dir) if report_dir else REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict = {}

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def run(self) -> dict:
        """
        Execute the benchmark and compute latency statistics.

        Returns
        -------
        dict
            Benchmark results: mean_ms, median_ms, p95_ms, p99_ms,
            max_ms, min_ms, throughput_per_second, n_iterations.
        """
        logger.info(
            "Starting inference benchmark | iterations=%d", self.n_iterations
        )

        latencies_ms: list[float] = []
        n_rows = len(self.X_sample)

        # Warm-up run (not recorded) — ensures model weights are
        # in CPU cache and avoids cold-start bias.
        _ = self.model.predict_proba(self.X_sample.iloc[[0]])

        rng = np.random.default_rng(seed=42)

        for i in range(self.n_iterations):
            row_idx = int(rng.integers(0, n_rows))
            row = self.X_sample.iloc[[row_idx]]

            t0 = time.perf_counter()
            _ = self.model.predict_proba(row)
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1_000)

        latencies = np.array(latencies_ms)

        mean_ms = float(np.mean(latencies))
        throughput = 1_000 / mean_ms  # predictions per second

        self.results = {
            "n_iterations": self.n_iterations,
            "mean_ms": round(mean_ms, 4),
            "median_ms": round(float(np.median(latencies)), 4),
            "p50_ms": round(float(np.percentile(latencies, 50)), 4),
            "p95_ms": round(float(np.percentile(latencies, 95)), 4),
            "p99_ms": round(float(np.percentile(latencies, 99)), 4),
            "max_ms": round(float(np.max(latencies)), 4),
            "min_ms": round(float(np.min(latencies)), 4),
            "throughput_per_second": round(throughput, 2),
        }

        self._log_results()

        return self.results

    def save(self, path: Path | str | None = None) -> None:
        """
        Save benchmark results to a JSON file.

        Parameters
        ----------
        path : Path | str | None
            Destination path. Defaults to report_dir/benchmark_results.json.
        """
        if not self.results:
            raise RuntimeError("Call run() before save().")

        out = Path(path) if path else self.report_dir / "benchmark_results.json"
        save_json(self.results, out)
        logger.info("Benchmark results saved → %s", out)

    # --------------------------------------------------
    # Private helpers
    # --------------------------------------------------

    def _log_results(self) -> None:
        r = self.results
        logger.info("─" * 50)
        logger.info("Inference Benchmark Results")
        logger.info("─" * 50)
        logger.info("  Iterations          : %d", r["n_iterations"])
        logger.info("  Mean latency        : %.4f ms", r["mean_ms"])
        logger.info("  Median (p50)        : %.4f ms", r["median_ms"])
        logger.info("  p95 latency         : %.4f ms", r["p95_ms"])
        logger.info("  p99 latency         : %.4f ms", r["p99_ms"])
        logger.info("  Max latency         : %.4f ms", r["max_ms"])
        logger.info("  Throughput          : %.2f pred/sec", r["throughput_per_second"])
        logger.info("─" * 50)
