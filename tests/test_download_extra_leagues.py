import pytest

from scripts.download_extra_leagues import validate_csv


EXTRA_CSV = b"""Country,League,Season,Date,Time,Home,Away,HG,AG,Res
Argentina,Primera Division,2020,01/03/20,19:00,Team A,Team B,2,1,H
"""


def test_validates_extra_league_country_csv_schema():
    validate_csv(EXTRA_CSV, "ARG")


def test_rejects_non_csv_or_missing_extra_league_result_fields():
    with pytest.raises(ValueError, match="ARG extra-league CSV is missing columns"):
        validate_csv(b"<html>Access denied</html>", "ARG")
