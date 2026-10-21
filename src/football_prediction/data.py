
"""Validation for historical football match results."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class MatchResult:
    """A completed match result with its match date and competition."""

    match_date: date
    competition: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    def __post_init__(self) -> None:
        if not isinstance(self.match_date, date):
            raise TypeError("match_date must be a datetime.date")

        for field_name in ("competition", "home_team", "away_team"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if self.home_team.strip() == self.away_team.strip():
            raise ValueError("home_team and away_team must be different")

        for field_name in ("home_goals", "away_goals"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")
