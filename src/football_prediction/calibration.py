"""Leakage-aware multiclass temperature calibration for 1X2 forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, isfinite, log


class MulticlassTemperatureCalibrator:
    """Fit one softmax temperature from earlier out-of-sample predictions.

    Input rows are ``(probabilities, actual_class)`` with class order
    home/draw/away and labels 0/1/2. The fitted transform is suitable for 1X2
    outcome probabilities; it does not calibrate goal totals or scorelines.
    """

    def __init__(self, minimum_samples: int = 100) -> None:
        if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int):
            raise TypeError("minimum_samples must be an integer")
        if minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        self.minimum_samples = minimum_samples
        self.temperature: float | None = None

    def fit(self, rows: Sequence[tuple[Sequence[float], int]]) -> "MulticlassTemperatureCalibrator":
        if len(rows) < self.minimum_samples:
            raise ValueError(
                f"at least {self.minimum_samples} calibration samples are required"
            )
        normalized = [(self._validate(probabilities), actual) for probabilities, actual in rows]
        if any(isinstance(actual, bool) or actual not in (0, 1, 2) for _, actual in normalized):
            raise ValueError("actual class must be 0, 1, or 2")

        # Golden-section minimization in log-temperature space keeps the
        # temperature positive and avoids adding a numerical dependency.
        left, right = log(0.1), log(10.0)
        ratio = (5**0.5 - 1.0) / 2.0
        x1 = right - ratio * (right - left)
        x2 = left + ratio * (right - left)
        f1 = self._loss(normalized, exp(x1))
        f2 = self._loss(normalized, exp(x2))
        for _ in range(80):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - ratio * (right - left)
                f1 = self._loss(normalized, exp(x1))
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + ratio * (right - left)
                f2 = self._loss(normalized, exp(x2))
        self.temperature = exp((left + right) / 2.0)
        return self

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        if self.temperature is None:
            raise RuntimeError("calibrator has not been fitted")
        values = self._validate(probabilities)
        logits = [log(max(value, 1e-15)) / self.temperature for value in values]
        maximum = max(logits)
        scaled = [exp(value - maximum) for value in logits]
        total = sum(scaled)
        return [value / total for value in scaled]

    @staticmethod
    def _validate(probabilities: Sequence[float]) -> tuple[float, float, float]:
        if len(probabilities) != 3:
            raise ValueError("probabilities must contain home, draw, and away values")
        values = tuple(probabilities)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("probabilities must be finite numbers between 0 and 1")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to 1")
        return values

    @staticmethod
    def _loss(rows, temperature: float) -> float:
        total = 0.0
        for probabilities, actual in rows:
            logits = [log(max(value, 1e-15)) / temperature for value in probabilities]
            maximum = max(logits)
            denominator = maximum + log(sum(exp(value - maximum) for value in logits))
            total += denominator - logits[actual]
        return total / len(rows)
