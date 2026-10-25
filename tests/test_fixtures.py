from datetime import date
from pathlib import Path

import pytest

from football_prediction.fixtures import (
    historical_team_name,
    load_verified_fixtures,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_officially_sourced_october_25_schedule():
    fixtures = load_verified_fixtures(
        PROJECT_ROOT / "data" / "fixtures" / "epl_2026-10-25.csv",
        date(2026, 10, 25),
    )

    assert len(fixtures) == 5
    assert fixtures[0]["home_team"] == "Crystal Palace"
    assert fixtures[0]["away_team"] == "Newcastle United"
    assert all(fixture["source_url"].startswith("https://") for fixture in fixtures)


def test_fixture_loader_rejects_a_date_other_than_requested():
    path = PROJECT_ROOT / "data" / "fixtures" / "epl_2026-10-25.csv"

    with pytest.raises(ValueError, match="expected 2026-10-24"):
        load_verified_fixtures(path, date(2026, 10, 24))


def test_official_team_names_map_to_names_in_historical_csvs():
    assert historical_team_name("AFC Bournemouth") == "Bournemouth"
    assert historical_team_name("Brighton & Hove Albion") == "Brighton"
    assert historical_team_name("Hull City") == "Hull City"
