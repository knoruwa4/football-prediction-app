# Football Prediction App

An evidence based football prediction platform. The project will develop statistical models, preserve forecasts, and evaluate them against results. Predictions represent uncertainty; they are not guarantees.

## Current status

The repository is being initialized. The first implementation stage will provide a small Python prediction engine with a transparent Poisson baseline. No historical dataset or live fixture integration is configured yet, so the engine will only produce probabilities from explicit expected goal inputs.

## Planned stack

- Prediction engine: Python
- Web: Next.js and TypeScript (later stage)
- Mobile: React Native, Expo, and TypeScript (later stage)
- Database: PostgreSQL, with Supabase under evaluation

Models will be compared using chronological backtesting before any change is promoted. Additional model types can be introduced behind the same interface when data and evaluation support them.
