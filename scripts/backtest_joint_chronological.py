
"""Chronological comparison of current and joint attack/defence models."""

import csv
from collections import defaultdict
from pathlib import Path

from football_prediction.football_data import load_football_data
from football_prediction.predictor import predict_match
from backtest_joint_model import JointAttackDefenceModel, outcome_of, probabilities, score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_PATH = ROOT / "reports" / "joint_chronological_comparison.csv"

MIN_HISTORY = 60
PRIOR_MATCHES = 50.0


def main():
    files = sorted(DATA_DIR.glob("epl_*.csv"))
    if not files:
        raise FileNotFoundError(f"No EPL CSV files in {DATA_DIR}")

    matches = []
    source_files = {}

    for path in files:
        for match in load_football_data(path):
            key = (match.match_date, match.home_team, match.away_team)
            if key in source_files:
                raise ValueError(f"Duplicate fixture found: {key}")
            source_files[key] = path.name
            matches.append(match)

    matches.sort(
        key=lambda m: (m.match_date, m.home_team, m.away_team)
    )

    # Group fixtures by date so results from the same date never enter
    # one another's training history.
    by_date = defaultdict(list)
    for match in matches:
        by_date[match.match_date].append(match)

    history = []
    rows = []
    skipped = 0
    dates = sorted(by_date)

    for index, match_date in enumerate(dates):
        fixtures = by_date[match_date]

        if len(history) < MIN_HISTORY:
            skipped += len(fixtures)
            history.extend(fixtures)
            continue

        print(
            f"Processing {match_date}: {len(fixtures)} fixtures "
            f"with {len(history)} historical matches"
        )

        # Fit once per date; all fixtures on the date share the same
        # strictly earlier training history.
        joint_model = JointAttackDefenceModel(history)

        for match in fixtures:
            try:
                current = predict_match(
                    history,
                    match.home_team,
                    match.away_team,
                    match.match_date,
                    prior_matches=PRIOR_MATCHES,
                    half_life_days=120.0,
                    allow_unknown_teams=True,
                )
                joint = joint_model.predict(
                    match.home_team, match.away_team
                )
            except (ValueError, KeyError) as error:
                print(
                    f"Skipping {match_date} {match.home_team} - "
                    f"{match.away_team}: {error}"
                )
                skipped += 1
                continue

            actual = outcome_of(match)
            rows.append({
                "date": match_date.isoformat(),
                "season_file": source_files[
                    (match.match_date, match.home_team, match.away_team)
                ],
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_outcome": ("H", "D", "A")[actual],
                "current_home": current.home_win,
                "current_draw": current.draw,
                "current_away": current.away_win,
                "joint_home": joint.home_win,
                "joint_draw": joint.draw,
                "joint_away": joint.away_win,
                "_actual": actual,
            })

        # Only add today's results after all predictions for today.
        history.extend(fixtures)

    if not rows:
        print("No matches were eligible for evaluation.")
        return

    def print_report(label, selected):
        print(f"\n{label}: {len(selected)} matches")

        for name, keys in (
            (
                "Current team-strength model",
                ("current_home", "current_draw", "current_away"),
            ),
            (
                "Joint attack/defence model",
                ("joint_home", "joint_draw", "joint_away"),
            ),
        ):
            values = [
                ([row[key] for key in keys], row["_actual"])
                for row in selected
            ]
            brier, loss = score(values)
            print(f"  {name}: Brier={brier:.4f}, log loss={loss:.4f}")

    print(f"Input seasons: {len(files)}")
    print(f"Total fixtures: {len(matches)}")
    print(f"Evaluated fixtures: {len(rows)}")
    print(f"Skipped fixtures: {skipped}")

    print_report("Overall", rows)

    by_season = defaultdict(list)
    for row in rows:
        by_season[row["season_file"]].append(row)

    for season in sorted(by_season):
        print_report(season, by_season[season])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [key for key in rows[0] if not key.startswith("_")]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in fields}
            for row in rows
        )

    print(f"\nReport saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
