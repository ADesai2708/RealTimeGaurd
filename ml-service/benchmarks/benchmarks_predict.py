"""
Benchmark the RealTimeGuard /predict API.

Measures:
- total requests
- successful requests
- failed requests
- requests per second
- average latency
- p50 latency
- p95 latency
- p99 latency
"""

from __future__ import annotations

import statistics
import time

import httpx


BASE_URL = "http://127.0.0.1:8000"

NUM_REQUESTS = 100

TRANSACTION = {
    "step": 120,
    "type": "TRANSFER",
    "amount": 7500.0,
    "oldbalanceOrg": 15000.0,
    "newbalanceOrig": 7500.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 8500.0,
}


def percentile(values: list[float], percentage: float) -> float:
    """Calculate a percentile using linear interpolation."""

    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = (len(sorted_values) - 1) * (percentage / 100)

    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)

    weight = index - lower

    return (
        sorted_values[lower]
        + weight * (sorted_values[upper] - sorted_values[lower])
    )


def main() -> None:
    latencies: list[float] = []

    successful = 0
    failed = 0

    print("=" * 60)
    print("RealTimeGuard API Benchmark")
    print("=" * 60)
    print(f"Target:       {BASE_URL}/predict")
    print(f"Requests:     {NUM_REQUESTS}")
    print()

    with httpx.Client(timeout=10.0) as client:

        # Warm-up request
        print("Running warm-up request...")

        response = client.post(
            f"{BASE_URL}/predict",
            json=TRANSACTION,
        )

        if response.status_code != 200:
            print(
                f"Warm-up failed with status "
                f"{response.status_code}: {response.text}"
            )
            return

        print("Warm-up successful.")
        print()

        # Actual benchmark
        print("Running benchmark...")

        start_time = time.perf_counter()

        for _ in range(NUM_REQUESTS):

            request_start = time.perf_counter()

            try:
                response = client.post(
                    f"{BASE_URL}/predict",
                    json=TRANSACTION,
                )

                request_latency = (
                    time.perf_counter() - request_start
                ) * 1000

                if response.status_code == 200:
                    successful += 1
                    latencies.append(request_latency)
                else:
                    failed += 1

            except Exception:
                failed += 1

        total_time = time.perf_counter() - start_time

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    if not latencies:
        print("No successful requests were recorded.")
        return

    average_latency = statistics.mean(latencies)

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    requests_per_second = successful / total_time

    print()
    print("=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    print(f"Total requests:       {NUM_REQUESTS}")
    print(f"Successful requests:  {successful}")
    print(f"Failed requests:      {failed}")
    print(f"Total time:           {total_time:.4f} sec")
    print(f"Requests/sec:         {requests_per_second:.2f}")
    print()
    print(f"Average latency:      {average_latency:.2f} ms")
    print(f"P50 latency:          {p50:.2f} ms")
    print(f"P95 latency:          {p95:.2f} ms")
    print(f"P99 latency:          {p99:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()