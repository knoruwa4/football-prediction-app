from datetime import date
from pathlib import Path

import pytest

from football_prediction.fixtures import (
    historical_team_name,
    load_verified_fixtures,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_officially_sourced_fixture(tmp_path):
    fixture_path = tmp_path / "fixtures.csv"
    fixture_path.write_text(
        "match_date,kickoff_utc,home_team,away_team,source_url\n"
        "2026-11-06,20:00,Everton,Coventry City,https://example.com/fixtures\n",
        encoding="utf-8",
    )
    fixtures = load_verified_fixtures(
        fixture_path,
        date(2026, 11, 6),
    )

    assert len(fixtures) == 1
    assert fixtures[0]["home_team"] == "Everton"
    assert fixtures[0]["away_team"] == "Coventry City"
    assert all(fixture["source_url"].startswith("https://") for fixture in fixtures)


def test_fixture_loader_rejects_a_date_other_than_requested():
    path = PROJECT_ROOT / "data" / "fixtures" / "epl_2026-10-25.csv"

    with pytest.raises(ValueError, match="expected 2026-10-24"):
        load_verified_fixtures(path, date(2026, 10, 24))


def test_official_team_names_map_to_names_in_historical_csvs():
    assert historical_team_name("AFC Bournemouth") == "Bournemouth"
    assert historical_team_name("Brighton & Hove Albion") == "Brighton"
    assert historical_team_name("Hull City") == "Hull"
    assert historical_team_name("Manchester City") == "Man City"
    assert historical_team_name("Nottingham Forest") == "Nott'm Forest"
    assert historical_team_name("Coventry City") == "Coventry"
    assert historical_team_name("Tottenham Hotspur") == "Tottenham"
    assert historical_team_name("Atletico Madrid") == "Atlético Madrid"
    assert historical_team_name("Bayern Munich") == "Bayern München"
    assert historical_team_name("Barcelona") == "FC Barcelona"
    assert historical_team_name("Bodoe/Glimt") == "Bodø/Glimt"
    assert historical_team_name("Man Utd") == "Man United"
