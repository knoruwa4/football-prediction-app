# Initial model assumptions

The first engine is an independent Poisson goal model. It accepts home and away expected goals and produces probabilities for 1X2, double chance, goal totals, and both teams to score.

## Scope and limitations

- The engine does not estimate expected goals from team data yet.
- Home and away scores are assumed independent; real match scores can be correlated.
- A single Poisson rate may not fit every team or competition.
- Score enumeration leaves less than 1e-12 probability mass outside each team's score range; inputs are limited to 0–20 expected goals.
- Outputs are model probabilities, not confidence scores, betting advice, or guarantees.
- No real fixture predictions should be presented until dated fixtures and validated expected-goal inputs are supplied.

Alternative models can implement the PredictionModel interface. Compare them with chronological backtests, calibration, Brier score, and log loss before adoption.
