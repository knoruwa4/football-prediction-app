import pytest

from football_prediction.league_data import (
    load_extra_league_history,
    load_league_history,
    load_main_league_histories,
    season_file,
)


CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR
SP1,15/08/2020,18:00,Team A,Team B,2,0,H
SP1,16/08/2020,18:00,Team C,Team D,1,1,D
"""


def test_loads_existing_season_file_for_one_competition(tmp_path):
    path = season_file(tmp_path, 2020, "SP1")
    path.parent.mkdir(parents=True)
    path.write_text(CSV, encoding="utf-8")

    matches = load_league_history(tmp_path, "SP1", [2020])

    assert len(matches) == 2
    assert {match.competition for match in matches} == {"La Liga"}


def test_rejects_a_missing_requested_season(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing La Liga season file"):
        load_league_history(tmp_path, "SP1", [2020])


def test_loads_leagues_as_separate_histories_and_reuses_year_iterable(tmp_path):
    for code, goals in (("SP1", "2,0"), ("I1", "0,2")):
        path = season_file(tmp_path, 2020, code)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CSV.replace("SP1,", f"{code},").replace("2,0,H", goals + ",H"), encoding="utf-8")

    histories = load_main_league_histories(
        tmp_path, ["SP1", "I1"], (year for year in [2020])
    )

    assert {match.competition for match in histories["SP1"]} == {"La Liga"}
    assert {match.competition for match in histories["I1"]} == {"Serie A"}
    assert histories["SP1"][0].home_goals == 2
    assert histories["I1"][0].home_goals == 0


def test_rejects_duplicate_fixtures_across_season_files(tmp_path):
    for year in (2020, 2021):
        path = season_file(tmp_path, year, "SP1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CSV, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate La Liga fixture"):
        load_league_history(tmp_path, "SP1", [2020, 2021])


def test_extra_league_loader_filters_season_and_unlisted_league_rows(tmp_path):
    path = tmp_path / "extra" / "ARG.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Argentina,Copa De La Liga Profesional,2020,01/10/20,18:00,Cup A,Cup B,2,0,H\n"
        "Argentina,Liga Profesional,2020,01/11/20,18:00,League A,League B,1,1,D\n"
        "Argentina,Liga Profesional,2021,01/03/21,18:00,League C,League D,0,2,A\n",
        encoding="utf-8",
    )

    matches = load_extra_league_history(tmp_path, "ARG", [2020])

    assert len(matches) == 1
    assert matches[0].competition == "Argentina Primera Division"
    assert (matches[0].home_team, matches[0].away_team) == ("League A", "League B")
    assert matches[0].match_date.isoformat() == "2020-11-01"


def test_extra_league_loader_reads_slash_seasons_and_filters_lower_division(tmp_path):
    path = tmp_path / "extra" / "SWZ.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Switzerland,Challenge League,2020/2021,01/10/20,18:00,Lower A,Lower B,2,0,H\n"
        "Switzerland,Super League,2020/2021,02/10/20,18:00,Top A,Top B,1,1,D\n",
        encoding="utf-8",
    )

    matches = load_extra_league_history(tmp_path, "SWZ", [2020])

    assert len(matches) == 1
    assert (matches[0].home_team, matches[0].away_team) == ("Top A", "Top B")
