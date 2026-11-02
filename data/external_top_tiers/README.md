# Supplemental competition data

This directory contains completed results selected from the schochastics/football-data results dataset. Source: https://github.com/schochastics/football-data (ODC-By Attribution license). The source README describes its coverage as top-tier domestic leagues and international tournaments; its competition field identifies domestic leagues by country. Only countries not already represented by the project’s existing league CSVs were imported.

Each CSV uses the project’s Football-Data-compatible columns (Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR) plus Season. Domestic records are limited to the top-tier national competition for the country. International records are limited to UEFA Champions League, UEFA Europa League, and AFC Champions League. Only records with full_time=F (ended in 90 minutes) were retained; extra-time and penalty-shootout records are excluded because their scores are not regulation-time results. Exact duplicate rows and records listing the same team as both home and away were removed.

Coverage: calendar-year leagues use 2018–2023; season-based leagues and UEFA competitions use 2017/18–2022/23; AFC Champions League uses 2018–2023. The source ends in 2023, so it does not include later results. Some seasons are disrupted or incomplete in the source (for example, Jamaica 2020/21 has five matches and Indonesia 2020 has 26). This data does not cover lower divisions, women’s leagues, reserve leagues, or domestic cups. UEFA Conference League is excluded because the source only has three seasons.

These files can be loaded with load_football_data(path, expected_division_code=...). They are kept separate from data YYYY/ so they do not silently alter the existing training data selection.
