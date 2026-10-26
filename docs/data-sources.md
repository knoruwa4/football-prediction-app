# Data sources and current coverage

## Football-Data.co.uk results

The project uses Football-Data.co.uk downloadable CSV results. The [main-league files](https://football-data.co.uk/englandm.php) are organized by season and division; the [extra-league index](https://football-data.co.uk/all_new_data.php) links one all-seasons CSV for each country. The provider describes the CSV data as free. Its separately marketed API is not used.

### Main leagues

The local workspace contains the provider's 22 main divisions for each season start year 2020–2026 in `data/data YYYY/CODE.csv` (154 files).

| Country | Division codes and competitions |
| --- | --- |
| England | `E0` Premier League, `E1` Championship, `E2` League One, `E3` League Two, `EC` National League |
| Scotland | `SC0` Premiership, `SC1` Championship, `SC2` League One, `SC3` League Two |
| Germany | `D1` Bundesliga, `D2` 2. Bundesliga |
| Italy | `I1` Serie A, `I2` Serie B |
| Spain | `SP1` La Liga, `SP2` La Liga 2 |
| France | `F1` Ligue 1, `F2` Ligue 2 |
| Netherlands | `N1` Eredivisie |
| Belgium | `B1` Pro League |
| Portugal | `P1` Primeira Liga |
| Turkey | `T1` Super Lig |
| Greece | `G1` Super League |

The audit validated 48,219 completed results across these files. Seasons 2020/21–2025/26 are complete; 2026/27 is an in-progress snapshot.

### Extra leagues

All 16 all-seasons country CSVs are downloaded into `data/extra/CODE.csv` and integrated into the history loader.

| Country | Code | Selected competition |
| --- | --- | --- |
| Argentina | `ARG` | Liga Profesional (the provider's Primera Division country file) |
| Austria | `AUT` | Bundesliga |
| Brazil | `BRA` | Serie A |
| China | `CHN` | Super League |
| Denmark | `DNK` | Superliga |
| Finland | `FIN` | Veikkausliiga |
| Ireland | `IRL` | Premier Division |
| Japan | `JPN` | J1 League |
| Mexico | `MEX` | Liga MX |
| Norway | `NOR` | Eliteserien |
| Poland | `POL` | Ekstraklasa |
| Romania | `ROU` | Superliga |
| Russia | `RUS` | Premier League |
| Sweden | `SWE` | Allsvenskan |
| Switzerland | `SWZ` | Super League |
| USA | `USA` | MLS |

The extra-league loader filters the `League` field to the selected competition and reads the provider's `Season`, `Date`, `Home`, `Away`, `HG`, and `AG` fields. Season labels can be calendar years (`2020`) or split years (`2020/2021`). The loader selects by the first year in the provider label; calendar-year leagues therefore represent calendar-year seasons rather than European summer-to-summer seasons.

The audit found 28,919 completed matches in the selected extra-league histories from provider seasons starting 2020 through 2026. Argentina has no `Liga Profesional` rows for provider season 2020; its rows that year are labeled `Copa De La Liga Profesional` and are excluded. Switzerland's `Challenge League` rows are excluded. This keeps separate competitions from being combined in a single model history.

## Model data and limits

- Match result models use only match date, home and away team, and full-time goals. Odds, half-time scores, and other match statistics are not currently model features.
- Each forecast is fitted on matches from its own competition and only results before the forecast date. Matches played on that same date are excluded.
- Data provider coverage, season lengths, and division formats vary. Current 2026/27 files may be incomplete or updated later; rerun `python scripts/inspect_league_data.py` before prediction or evaluation runs.
- Some clubs change names across seasons. The model treats source labels as team identities; aliases need to be checked against actual provider names before being added.
- Fixture schedules are separate from results. Each fixture must be verified and stored with its source URL before generating a forecast.
- Historical odds must not be model predictors unless their timestamp is known to precede the forecast time.
