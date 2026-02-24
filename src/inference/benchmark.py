"""Inference performance benchmark script."""

from __future__ import annotations

import argparse
import time


def benchmark_stub(iterations: int, batch_size: int) -> None:
    start = time.perf_counter()
    for _ in range(iterations):
        _ = [0] * batch_size
    elapsed = time.perf_counter() - start
    print({
        "iterations": iterations,
        "batch_size": batch_size,
        "latency_ms": (elapsed / iterations) * 1000,
        "throughput": iterations * batch_size / elapsed,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    benchmark_stub(args.iterations, args.batch_size)
