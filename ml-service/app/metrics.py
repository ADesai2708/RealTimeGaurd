"""
app/metrics.py

Lightweight in-process inference metrics for the RealTimeGuard service.

Design decisions:
-----------------
Thread safety: All mutations use a ``threading.Lock``. FastAPI runs
    request handlers in a thread pool, so shared mutable state requires
    synchronization.  This is sufficient for a single Uvicorn process.

Bounded latency buffer: A ``collections.deque(maxlen=10_000)`` keeps the
    last 10,000 latency samples.  Older values are silently dropped when
    the buffer is full, preventing unbounded memory growth in long-running
    deployments.

Percentile computation: ``numpy.percentile`` on the latency buffer is done
    lazily (only when ``/metrics`` is called), not on every request.

⚠ Limitation — single-process only:
    These metrics are stored in the process's memory.  If you run multiple
    Uvicorn workers (``--workers 4``) or multiple container replicas, each
    process maintains its own independent counter set.  ``/metrics`` will
    only reflect traffic handled by the process that received the request.

    Aggregated metrics across workers require an external store such as
    Redis, Prometheus, or a time-series database.  That belongs in a later
    observability phase.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np


# Maximum number of latency samples to retain in memory.
_LATENCY_BUFFER_SIZE = 10_000


@dataclass
class InferenceMetrics:
    """
    Thread-safe, in-process inference counters and latency statistics.

    Instantiate once and store on ``app.state`` during startup.
    """

    # ------------------------------------------------------------------
    # Internal state — never access directly from outside this class
    # ------------------------------------------------------------------
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _latency_buffer: deque = field(
        default_factory=lambda: deque(maxlen=_LATENCY_BUFFER_SIZE), repr=False
    )

    # Prediction counts
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0

    # Label counts
    fraud_predictions: int = 0
    legitimate_predictions: int = 0

    # Decision counts
    approve_decisions: int = 0
    review_decisions: int = 0
    block_decisions: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        *,
        latency_ms: float,
        is_fraud: bool,
        decision: str,
        success: bool = True,
    ) -> None:
        """
        Record the result of one prediction request.

        Parameters
        ----------
        latency_ms : float
            Predictor latency in milliseconds.
        is_fraud : bool
            True if the model predicted FRAUD.
        decision : str
            One of "APPROVE", "REVIEW", "BLOCK".
        success : bool
            False if the prediction pipeline raised an exception.
        """
        with self._lock:
            self.total_predictions += 1

            if success:
                self.successful_predictions += 1
                self._latency_buffer.append(latency_ms)

                if is_fraud:
                    self.fraud_predictions += 1
                else:
                    self.legitimate_predictions += 1

                decision_upper = decision.upper()
                if decision_upper == "APPROVE":
                    self.approve_decisions += 1
                elif decision_upper == "REVIEW":
                    self.review_decisions += 1
                elif decision_upper == "BLOCK":
                    self.block_decisions += 1
            else:
                self.failed_predictions += 1

    def record_failure(self) -> None:
        """Convenience method to increment failure counters only."""
        with self._lock:
            self.total_predictions += 1
            self.failed_predictions += 1

    def to_dict(self) -> dict:
        """
        Return a snapshot of current metrics as a plain dict.

        Percentiles are computed from the in-memory latency buffer.
        Returns 0.0 for all latency statistics when no successful
        predictions have been recorded yet.
        """
        with self._lock:
            buf = list(self._latency_buffer)  # snapshot under lock
            total = self.total_predictions
            successful = self.successful_predictions
            failed = self.failed_predictions
            fraud = self.fraud_predictions
            legitimate = self.legitimate_predictions
            approve = self.approve_decisions
            review = self.review_decisions
            block = self.block_decisions

        if buf:
            arr = np.array(buf, dtype=np.float64)
            avg_ms = float(np.mean(arr))
            p50_ms = float(np.percentile(arr, 50))
            p95_ms = float(np.percentile(arr, 95))
            p99_ms = float(np.percentile(arr, 99))
        else:
            avg_ms = p50_ms = p95_ms = p99_ms = 0.0

        return {
            "total_predictions": total,
            "successful_predictions": successful,
            "failed_predictions": failed,
            "average_latency_ms": round(avg_ms, 4),
            "p50_latency_ms": round(p50_ms, 4),
            "p95_latency_ms": round(p95_ms, 4),
            "p99_latency_ms": round(p99_ms, 4),
            "fraud_predictions": fraud,
            "legitimate_predictions": legitimate,
            "approve_decisions": approve,
            "review_decisions": review,
            "block_decisions": block,
        }
