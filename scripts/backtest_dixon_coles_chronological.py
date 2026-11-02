"""Season-held-out comparison of joint Poisson and Dixon–Coles probabilities.

Each test season is predicted using only earlier seasons. No results from a
test season enter that season's forecasts. This conservative protocol avoids
repeated expensive refits and should not be confused with weekly live updates.
"""

from __future__ import annotations

import csv
from math import log
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_prediction.dixon_coles import DixonColesModel
from football_prediction.elo_model import EloAdjustedPoissonModel
from football_prediction.football_data import load_football_data
from football_prediction.joint_model import JointAttackDefenceModel
from football_prediction.calibration import MulticlassTemperatureCalibrator


DATA_DIR = ROOT / "data" / "raw"
REPORT_PATH = ROOT / "reports" / "dixon_coles_chronological_comparison.csv"
MIN_HISTORY = 60
RHO = -0.05  # Prespecified candidate; do not tune on this held-out report.
EPSILON = 1e-15


def _actual(match) -> int:
    if match.home_goals > match.away_goals:
        return 0
    if match.home_goals == match.away_goals:
        return 1
    return 2


def _scores(rows) -> tuple[float, float]:
    count = len(rows)
    brier = sum(
        sum((p - float(index == actual)) ** 2 for index, p in enumerate(probs))
        for probs, actual in rows
    ) / count
    log_loss = sum(-log(max(probs[actual], EPSILON)) for probs, actual in rows) / count
    return brier, log_loss


def main() -> None:
    files = sorted(DATA_DIR.glob("epl_*.csv"))
    if not files:
        raise FileNotFoundError(f"No EPL CSV files in {DATA_DIR}")

    seasons = {path.name: load_football_data(path) for path in files}
    history = []
    rows = []
    skipped_seasons = []
    calibration_history = {"joint": [], "dixon_coles": [], "elo": []}

    for season_file in sorted(seasons):
        fixtures = seasons[season_file]
        if len(history) < MIN_HISTORY:
            skipped_seasons.append(season_file)
            history.extend(fixtures)
            continue

        model = JointAttackDefenceModel(history)
        dixon_coles = DixonColesModel(RHO)
        elo = EloAdjustedPoissonModel(history)
        calibrators = {
            name: MulticlassTemperatureCalibrator().fit(samples)
            for name, samples in calibration_history.items()
            if len(samples) >= 100
        }
        season_calibration_rows = {"joint": [], "dixon_coles": [], "elo": []}
        for match in fixtures:
            joint = model.predict(match.home_team, match.away_team)
            corrected = dixon_coles.predict(joint.home_xg, joint.away_xg)
            elo_prediction = elo.predict(match.home_team, match.away_team)
            actual = _actual(match)
            predictions = {
                "joint": [joint.home_win, joint.draw, joint.away_win],
                "dixon_coles": [corrected.home_win, corrected.draw, corrected.away_win],
                "elo": [elo_prediction.home_win, elo_prediction.draw, elo_prediction.away_win],
            }
            row = {
                "date": match.match_date.isoformat(),
                "season_file": season_file,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_outcome": ("H", "D", "A")[actual],
                "joint_home": joint.home_win,
                "joint_draw": joint.draw,
                "joint_away": joint.away_win,
                "dixon_coles_home": corrected.home_win,
                "dixon_coles_draw": corrected.draw,
                "dixon_coles_away": corrected.away_win,
                "elo_home": elo_prediction.home_win,
                "elo_draw": elo_prediction.draw,
                "elo_away": elo_prediction.away_win,
            }
            for name, probabilities in predictions.items():
                season_calibration_rows[name].append((probabilities, actual))
                calibrator = calibrators.get(name)
                row[f"{name}_calibration_temperature"] = (
                    calibrator.temperature if calibrator else ""
                )
                calibrated = calibrator.transform(probabilities) if calibrator else None
                for index, side in enumerate(("home", "draw", "away")):
                    row[f"{name}_calibrated_{side}"] = (
                        calibrated[index] if calibrated else ""
                    )
            rows.append(row)
        # Calibration examples from this season become available only to the
        # next held-out season.
        for name, samples in season_calibration_rows.items():
            calibration_history[name].extend(samples)
        # Add a season's scores only after its forecasts are complete.
        history.extend(fixtures)
        print(f"Processed {season_file}: {len(fixtures)} fixtures using {len(history) - len(fixtures)} earlier matches")

    if not rows:
        raise ValueError("No matches were eligible for evaluation")

    model_columns = (
        ("Joint Poisson", "joint"),
        ("Dixon–Coles", "dixon_coles"),
        ("Elo-adjusted Poisson", "elo"),
    )
    for label, prefix in model_columns:
        values = [
            ([row[f"{prefix}_{side}"] for side in ("home", "draw", "away")],
             ("H", "D", "A").index(row["actual_outcome"]))
            for row in rows
        ]
        brier, log_loss = _scores(values)
        print(f"{label}: Brier={brier:.4f}, log loss={log_loss:.4f}")

    for season_file in sorted({row["season_file"] for row in rows}):
        season_rows = [row for row in rows if row["season_file"] == season_file]
        print(f"{season_file} ({len(season_rows)} matches)")
        for label, prefix in model_columns:
            values = [
                ([row[f"{prefix}_{side}"] for side in ("home", "draw", "away")],
                 ("H", "D", "A").index(row["actual_outcome"]))
                for row in season_rows
            ]
            brier, log_loss = _scores(values)
            print(f"  {label}: Brier={brier:.4f}, log loss={log_loss:.4f}")

    common_rows = [row for row in rows if row["joint_calibration_temperature"] != ""]
    if common_rows:
        print(f"\nTemperature calibration, {len(common_rows)} later-season forecasts:")
        for label, prefix in model_columns:
            raw = [
                ([row[f"{prefix}_{side}"] for side in ("home", "draw", "away")],
                 ("H", "D", "A").index(row["actual_outcome"]))
                for row in common_rows
            ]
            calibrated = [
                ([row[f"{prefix}_calibrated_{side}"] for side in ("home", "draw", "away")],
                 ("H", "D", "A").index(row["actual_outcome"]))
                for row in common_rows
            ]
            raw_brier, raw_loss = _scores(raw)
            cal_brier, cal_loss = _scores(calibrated)
            print(f"  {label}: raw Brier={raw_brier:.4f}, log loss={raw_loss:.4f}; calibrated Brier={cal_brier:.4f}, log loss={cal_loss:.4f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluated {len(rows)} matches; skipped seasons without prior history: {', '.join(skipped_seasons)}")
    print(f"Fixed rho={RHO}; report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
