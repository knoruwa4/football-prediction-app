# Football Prediction App

An evidence based football prediction platform. The project will develop statistical models, preserve forecasts, and evaluate them against results. Predictions represent uncertainty; they are not guarantees.

## Current status

The project includes Football-Data.co.uk Premier League results, independent and team-strength Poisson models, an experimental joint attack/defence model, batch predictions, chronological backtests, CSV forecast tracking, and a read-only local JSON API. Forecasts are recorded with a timestamp and model label, then evaluated when matching completed results are available. No live fixture provider is configured.

## Local API

Generate and save predictions with `python scripts/predict_fixtures.py`, then start the API with `python scripts/serve_api.py`. The server uses only Python's standard library and binds to `127.0.0.1:8000` by default. Set `FOOTBALL_API_HOST` or `FOOTBALL_API_PORT` to change the bind address or port.

- `GET /health` — service status.
- `GET /api/predictions` — saved forecasts.
- `GET /api/predictions?status=pending` — forecasts awaiting a result.
- `GET /api/predictions?match_date=YYYY-MM-DD` — forecasts for one match date.
- `GET /api/performance` — evaluated count, pending count, outcome accuracy, and multiclass Brier score, calculated from the ledger.

The API is read-only and intended for local development. Evaluate saved forecasts with `python scripts/evaluate_predictions.py` before requesting updated performance metrics.

## Planned stack

- Prediction engine: Python
- Web: Next.js and TypeScript (later stage)
- Mobile: React Native, Expo, and TypeScript (later stage)
- Database: PostgreSQL, with Supabase under evaluation

Models will be compared using chronological backtesting before any change is promoted. Additional model types can be introduced behind the same interface when data and evaluation support them.
