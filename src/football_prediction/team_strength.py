
"""Estimate expected goals from recency-weighted historical results."""

from dataclasses import dataclass
from datetime import date
from math import isfinite

from football_prediction.data import MatchResult


@dataclass(frozen=True)
class ExpectedGoals:
    """Estimated goals for the home and away teams."""

    home_xg: float
    away_xg: float


class TeamStrengthModel:
    """Estimate expected goals using recency-weighted team strengths."""

    name = "recency_weighted_team_strength"

    def __init__(
        self,
        matches: list[MatchResult],
        prior_matches: float = 25.0,
        half_life_days: float = 120.0,
        allow_unknown_teams: bool = False,
    ) -> None:
        if not matches:
            raise ValueError("matches must not be empty")

        for name, value in (
            ("prior_matches", prior_matches),
            ("half_life_days", half_life_days),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and greater than 0"
                )

        if not isinstance(allow_unknown_teams, bool):
            raise TypeError("allow_unknown_teams must be a boolean")

        competitions = {match.competition for match in matches}
        if len(competitions) != 1:
            raise ValueError("all matches must belong to one competition")

        self.competition = matches[0].competition
        self.latest_match_date = max(
            match.match_date for match in matches
        )
        self.half_life_days = float(half_life_days)
        self.allow_unknown_teams = allow_unknown_teams

        self.teams = {
            team
            for match in matches
            for team in (match.home_team, match.away_team)
        }

        # A match half a half-life old receives half as much weight.
        weighted_matches = []
        for match in matches:
            age_days = (
                self.latest_match_date - match.match_date
            ).days
            weight = 0.5 ** (
                age_days / self.half_life_days
            )
            weighted_matches.append((match, weight))

        total_weight = sum(
            weight for _, weight in weighted_matches
        )

        self.league_home_goals = sum(
            match.home_goals * weight
            for match, weight in weighted_matches
        ) / total_weight

        self.league_away_goals = sum(
            match.away_goals * weight
            for match, weight in weighted_matches
        ) / total_weight

        if (
            self.league_home_goals <= 0
            or self.league_away_goals <= 0
        ):
            raise ValueError(
                "league home and away goal averages must both be positive"
            )

        # [weighted games, weighted goals scored, weighted goals conceded]
        home_stats = {
            team: [0.0, 0.0, 0.0] for team in self.teams
        }
        away_stats = {
            team: [0.0, 0.0, 0.0] for team in self.teams
        }

        for match, weight in weighted_matches:
            home = home_stats[match.home_team]
            home[0] += weight
            home[1] += match.home_goals * weight
            home[2] += match.away_goals * weight

            away = away_stats[match.away_team]
            away[0] += weight
            away[1] += match.away_goals * weight
            away[2] += match.home_goals * weight

        def shrunk_ratio(
            stats: list[float],
            league_average: float,
        ) -> float:
            weighted_games, goals, _ = stats
            rate = (
                goals + prior_matches * league_average
            ) / (weighted_games + prior_matches)
            return rate / league_average

        self.home_attack = {
            team: shrunk_ratio(stats, self.league_home_goals)
            for team, stats in home_stats.items()
        }
        self.home_defence = {
            team: shrunk_ratio(stats, self.league_away_goals)
            for team, stats in home_stats.items()
        }
        self.away_attack = {
            team: shrunk_ratio(stats, self.league_away_goals)
            for team, stats in away_stats.items()
        }
        self.away_defence = {
            team: shrunk_ratio(stats, self.league_home_goals)
            for team, stats in away_stats.items()
        }

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: date,
    ) -> ExpectedGoals:
        """Estimate goals for a fixture after the training data ends."""

        if not isinstance(match_date, date):
            raise TypeError("match_date must be a datetime.date")

        if match_date <= self.latest_match_date:
            raise ValueError(
                "match_date must be after the latest training match"
            )

        if not isinstance(home_team, str) or not home_team.strip():
            raise ValueError("home_team must be a non-empty string")

        if not isinstance(away_team, str) or not away_team.strip():
            raise ValueError("away_team must be a non-empty string")

        if home_team.strip() == away_team.strip():
            raise ValueError(
                "home_team and away_team must be different"
            )

        home_team = home_team.strip()
        away_team = away_team.strip()

        unknown = {home_team, away_team} - self.teams
        if unknown and not self.allow_unknown_teams:
            raise ValueError(
                f"unknown team(s) in training data: "
                f"{', '.join(sorted(unknown))}"
            )

        # Unknown teams receive neutral attack and defence multipliers.
        home_attack = self.home_attack.get(home_team, 1.0)
        home_defence = self.home_defence.get(home_team, 1.0)
        away_attack = self.away_attack.get(away_team, 1.0)
        away_defence = self.away_defence.get(away_team, 1.0)

        home_xg = (
            self.league_home_goals
            * home_attack
            * away_defence
        )
        away_xg = (
            self.league_away_goals
            * away_attack
            * home_defence
        )

        return ExpectedGoals(
            home_xg=home_xg,
            away_xg=away_xg,
        )
