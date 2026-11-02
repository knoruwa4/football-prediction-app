# Football Prediction App

An evidence based football prediction platform. It develops statistical models, preserves forecasts, and evaluates them against completed results. Predictions represent uncertainty; they are not guarantees.

## Current status

The workspace contains Football-Data.co.uk results for all 22 divisions in the provider's Main Leagues list, across season folders 2020/21 through partial 2026/27, plus the 16 Extra League country CSVs. The loaders filter histories by competition and season while keeping each league separate. The 2020 Argentina Liga Profesional is absent from the file; its 2020 rows are labeled as a separate cup competition and are excluded from the league model. It also contains 22 supplemental competition histories in data/external_top_tiers/ (19 top-tier domestic competitions and three international club competitions), mapped to the same prediction interface. These supplemental files are loaded separately and are not mixed into the original season folders.

The project has independent and team-strength Poisson models, a joint attack/defence model, experimental Dixon–Coles and Elo-adjusted Poisson candidates, a 1X2 temperature-calibration experiment, batch predictions, chronological backtests, CSV forecast tracking, a local JSON API, and a Next.js dashboard. The dashboard can select a model candidate when generating a forecast. Joint attack/defence Poisson remains the default; no candidate is promoted automatically. Histories are isolated by competition. No live fixture provider is configured; the app requires future fixture CSV rows with HTTPS source URLs.

## Local league data

Main-league files use the existing provider layout:

```text
data/data 2020/E0.csv   # Premier League
data/data 2020/E1.csv   # Championship
data/data 2020/SC0.csv  # Scottish Premiership
data/data 2020/D2.csv   # 2. Bundesliga
data/data 2020/SP1.csv  # La Liga
```

All 22 main-division files are present for start years 2020 through 2026. The 2026/27 season is incomplete. The 16 extra-league files are in `data/extra/`; calendar-year and split-season labels are mapped to their actual provider season start years. Run `python scripts/inspect_league_data.py` to audit coverage. See [docs/data-sources.md](docs/data-sources.md) for limitations.

To refresh the extra-league downloads, run `python scripts/download_extra_leagues.py`; existing files are kept. The prediction loader uses each source `Season` value and filters to the named competition. These leagues do not all use European summer-to-summer seasons.

To forecast sourced fixtures for a main division, pass its Football-Data code:

```powershell
python scripts/predict_fixtures.py --competition-code SP1 --match-date 2026-10-26 --fixture-file data/fixtures/laliga_2026-10-26.csv
```

The fixture CSV must include a source URL for each match. Use the matching `--competition-code` for a main division or extra-league country file. The app does not invent or fetch fixtures.

## Local API

Generate forecasts with `python scripts/predict_fixtures.py`, then start the API with `python scripts/serve_api.py`. It binds to `127.0.0.1:8000` by default. Set `FOOTBALL_API_HOST` or `FOOTBALL_API_PORT` to change that.

- `GET /health` — service status.
- `GET /api/predictions` — saved forecasts.
- `GET /api/predictions?status=pending` — forecasts awaiting a result.
- `GET /api/predictions?match_date=YYYY-MM-DD` — forecasts on one date.
- Add `&competition=La%20Liga` (or another catalogued competition) to filter either endpoint.
- `GET /api/performance` — evaluated count, pending count, outcome accuracy, and multiclass Brier score.
- `GET /api/fixtures?competition=Premier%20League` — upcoming fixtures from sourced CSVs in `data/fixtures`.
- `POST /api/predict` — forecast a catalogued fixture and save it before kickoff. Send `{"fixture_id":"…","model_type":"joint_attack_defence"}`. Supported model types are `joint_attack_defence`, `dixon_coles`, and `elo_poisson`; omitting `model_type` uses the existing joint model.

The API is intended for local development. Evaluate saved forecasts with `python scripts/evaluate_predictions.py` before requesting updated performance metrics. See [docs/model-assumptions.md](docs/model-assumptions.md) for candidate assumptions and the initial held-out comparison.

## Planned stack

- Prediction engine: Python
- Web: Next.js and TypeScript
- Mobile: React Native, Expo, and TypeScript (later stage)
- Database: PostgreSQL, with Supabase under evaluation

Models will be compared through chronological backtesting before any change is promoted.
