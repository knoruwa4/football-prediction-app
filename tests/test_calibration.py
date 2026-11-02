import pytest

from football_prediction.calibration import MulticlassTemperatureCalibrator


def test_temperature_calibrator_sharpens_underconfident_forecasts():
    rows = [([0.4, 0.3, 0.3], 0)] * 120
    calibrator = MulticlassTemperatureCalibrator().fit(rows)

    adjusted = calibrator.transform([0.4, 0.3, 0.3])
    assert calibrator.temperature < 1
    assert adjusted[0] > 0.4
    assert sum(adjusted) == pytest.approx(1)


def test_temperature_calibrator_softens_overconfident_wrong_forecasts():
    rows = [([0.8, 0.1, 0.1], 1)] * 120
    calibrator = MulticlassTemperatureCalibrator().fit(rows)

    adjusted = calibrator.transform([0.8, 0.1, 0.1])
    assert calibrator.temperature > 1
    assert adjusted[0] < 0.8


def test_temperature_calibrator_requires_enough_rows_and_fit_before_transform():
    calibrator = MulticlassTemperatureCalibrator(minimum_samples=2)
    with pytest.raises(ValueError, match="at least 2"):
        calibrator.fit([([0.5, 0.3, 0.2], 0)])
    with pytest.raises(RuntimeError, match="not been fitted"):
        calibrator.transform([0.5, 0.3, 0.2])


@pytest.mark.parametrize(
    "probabilities", [[0.5, 0.5], [0.5, 0.3, 0.3], [1.2, -0.1, -0.1]]
)
def test_temperature_calibrator_rejects_invalid_probabilities(probabilities):
    calibrator = MulticlassTemperatureCalibrator(minimum_samples=1)
    with pytest.raises(ValueError):
        calibrator.fit([(probabilities, 0)])
