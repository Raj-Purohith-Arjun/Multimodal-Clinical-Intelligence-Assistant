"""Entrypoint for structured evaluation."""

from __future__ import annotations

from src.evaluation.evaluator import Evaluator


def run_evaluation() -> None:
    evaluator = Evaluator(model=None)  # placeholder wiring for production checkpoint load.
    _ = evaluator


if __name__ == "__main__":
    run_evaluation()
