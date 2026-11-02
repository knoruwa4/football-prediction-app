# Initial model assumptions

The independent Poisson baseline accepts home and away expected goals and
produces probabilities for 1X2, double chance, goal totals, and both teams to
score. Team-strength and joint attack/defence models estimate those goal rates
from historical results.

## Scope and limitations

- The baseline Poisson model assumes independent home and away scores; the
  Dixon–Coles candidate adjusts the four low-score outcomes.
- Expected goals are estimated from full-time results, not shot-location xG
  data.
- A single Poisson rate may not fit every team or competition.
- Score enumeration leaves less than 1e-12 probability mass outside each team's score range; inputs are limited to 0–20 expected goals.
- Outputs are model probabilities, not confidence scores, betting advice, or guarantees.
- Predictions require sourced fixtures and sufficient earlier results for the
  selected competition.

Alternative models can implement the PredictionModel interface. Compare them with chronological backtests, calibration, Brier score, and log loss before adoption.

## Experimental candidates

`dixon_coles` applies the Dixon–Coles low-score correction to expected goals
from the existing joint attack/defence model. Its current candidate value is
`rho = -0.05`; this is fixed and is not fitted from the evaluation matches.
The model remains optional and is not the default.

`elo_poisson` fits Elo ratings from earlier match results, then uses the rating
difference to shift the joint model's expected goals. Elo uses a 60-point home
advantage and K=20. These are initial candidate settings, not tuned or
calibrated parameters. The model remains optional and is not the default.

The season-held-out comparison in
`reports/dixon_coles_chronological_comparison.csv` predicts each season from
prior seasons only. It does not update ratings or attack/defence parameters
during the held-out season. Results are an exploratory comparison, not a claim
that a candidate will improve future predictions. Current CSV result data has
aggregate shot counts but no shot-based expected-goal (xG) values; the displayed
expected goals are model estimates. An xG feature requires a separately sourced
dataset.

Initial EPL comparison (2021/22 through 2025/26, 1,900 fixtures; 2020/21
excluded as the initial history season):

| Candidate | Multiclass Brier | Log loss |
| --- | ---: | ---: |
| Joint attack/defence Poisson | 0.6289 | 1.0449 |
| Dixon–Coles (`rho = -0.05`) | 0.6293 | 1.0454 |
| Elo-adjusted joint Poisson | 0.6030 | 1.0083 |

The Elo-adjusted candidate scored better in each of the five held-out seasons
in this exploratory run. It is still not promoted: its K-factor and goal-rate
mapping have not been tuned on a separate validation set, the held-out tests
freeze model parameters for each entire season, and more leagues and rolling
date cutoffs should be checked before changing the default.

## Temperature calibration experiment

`MulticlassTemperatureCalibrator` calibrates 1X2 probabilities by fitting one
temperature to log loss. In the chronological report, the calibrator for each
held-out season uses only raw forecasts and actual outcomes from earlier
held-out seasons; at least 100 earlier forecasts are required. The current
experiment evaluates later-season 1X2 forecasts only. It does not calibrate
goal totals, BTTS, or scorelines, so it is not applied to live forecasts or
shown as a site model choice.

On the 1,520 forecasts with prior calibration examples:

| Model | Raw Brier | Calibrated Brier | Raw log loss | Calibrated log loss |
| --- | ---: | ---: | ---: | ---: |
| Joint attack/defence Poisson | 0.6282 | 0.6262 | 1.0441 | 1.0442 |
| Dixon–Coles | 0.6284 | 0.6267 | 1.0443 | 1.0436 |
| Elo-adjusted Poisson | 0.6009 | 0.6042 | 1.0054 | 1.0110 |

The changes are small and mixed across metrics. Calibration remains an
evaluation utility; these results do not support applying one shared
calibration transform to every market or model.
