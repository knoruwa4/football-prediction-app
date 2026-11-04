
"""Inspect season CSV files before multi-season backtesting."""

import csv
from pathlib import Path
from collections import Counter

from football_prediction.football_data import load_football_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

files = sorted(DATA_DIR.glob("epl_*.csv"))

if not files:
    raise FileNotFoundError(f"No epl_*.csv files found in {DATA_DIR}")

all_fixtures = []
print(f"Found {len(files)} season files\n")

for path in files:
    matches = load_football_data(path)
    dates = [match.match_date for match in matches]
    teams = {
        team
        for match in matches
        for team in (match.home_team, match.away_team)
    }
    fixture_keys = [
        (m.match_date, m.home_team, m.away_team)
        for m in matches
    ]
    duplicates = len(fixture_keys) - len(set(fixture_keys))

    print(
        f"{path.name}: {len(matches)} matches | "
        f"{len(teams)} teams | "
        f"{min(dates)} to {max(dates)} | "
        f"duplicate fixtures: {duplicates}"
    )
    all_fixtures.extend(
        (path.name, key) for key in fixture_keys
    )

cross_file_counts = Counter(key for _, key in all_fixtures)
cross_file_duplicates = [
    key for key, count in cross_file_counts.items() if count > 1
]

print(f"\nRepeated fixtures across files: {len(cross_file_duplicates)}")

if cross_file_duplicates:
    for fixture in cross_file_duplicates[:10]:
        sources = [
            filename
            for filename, key in all_fixtures
            if key == fixture
        ]
        print(f"  {fixture}: {sources}")
