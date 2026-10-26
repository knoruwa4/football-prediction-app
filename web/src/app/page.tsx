"use client";

import { Fragment, useCallback, useEffect, useMemo, useState } from "react";

type Prediction = {
  prediction_id: string;
  prediction_timestamp: string;
  match_date: string;
  competition?: string;
  kickoff_utc: string;
  home_team: string;
  away_team: string;
  home_team_history_available: boolean;
  away_team_history_available: boolean;
  model: string;
  model_version?: string;
  home_win_probability: number;
  draw_probability: number;
  away_win_probability: number;
  home_expected_goals: number;
  away_expected_goals: number;
  score_1: string;
  score_1_probability?: number;
  score_2: string;
  score_2_probability?: number;
  score_3: string;
  score_3_probability?: number;
  evaluation_status: "pending" | "evaluated";
};

type PredictionResponse = { count: number; predictions: Prediction[] };
type Performance = {
  evaluated_count: number;
  pending_count: number;
  outcome_accuracy: number | null;
  multiclass_brier_score: number | null;
};
type MarketTab = "result" | "goals" | "btts" | "combos" | "team-goals" | "handicap" | "scores";
type OddsFormat = "decimal" | "fractional";

const LEAGUES = [
  "Premier League", "Championship", "League One", "League Two", "National League",
  "Scottish Premiership", "Scottish Championship", "Scottish League One", "Scottish League Two",
  "Bundesliga", "2. Bundesliga", "Serie A", "Serie B", "La Liga", "La Liga 2", "Ligue 1", "Ligue 2",
  "Eredivisie", "Belgian Pro League", "Primeira Liga", "Turkish Super Lig", "Greek Super League",
  "Argentina Primera Division", "Austrian Bundesliga", "Brazilian Serie A", "Chinese Super League",
  "Danish Superliga", "Finnish Veikkausliiga", "Irish Premier Division", "Japanese J1 League",
  "Mexican Liga MX", "Norwegian Eliteserien", "Polish Ekstraklasa", "Romanian Liga 1",
  "Russian Premier League", "Swedish Allsvenskan", "Swiss Super League", "Major League Soccer",
];

const percent = (value: number | null | undefined) =>
  value == null ? "—" : `${(value * 100).toFixed(1)}%`;

function formatDate(value: string, options: Intl.DateTimeFormatOptions = {}) {
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "short", day: "numeric", month: "short", timeZone: "UTC", ...options,
  }).format(new Date(`${value}T12:00:00Z`));
}

function formatGenerated(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC", timeZoneName: "short",
  }).format(new Date(value));
}

function initials(name: string) {
  return name.split(/\s+/).filter(Boolean).map((part) => part[0]).join("").slice(0, 3).toUpperCase();
}

function factorial(n: number) {
  let result = 1;
  for (let i = 2; i <= n; i += 1) result *= i;
  return result;
}

function poissonProbability(goals: number, expectedGoals: number) {
  return Math.exp(-expectedGoals) * expectedGoals ** goals / factorial(goals);
}

function goalMarkets(prediction: Prediction) {
  const cap = 12;
  const matrix = Array.from({ length: cap + 1 }, (_, home) =>
    Array.from({ length: cap + 1 }, (_, away) =>
      poissonProbability(home, prediction.home_expected_goals) * poissonProbability(away, prediction.away_expected_goals),
    ),
  );
  const total = matrix.flat().reduce((sum, value) => sum + value, 0);
  const probability = (condition: (home: number, away: number) => boolean) =>
    matrix.reduce((sum, row, home) =>
      sum + row.reduce((rowSum, value, away) => rowSum + (condition(home, away) ? value : 0), 0), 0) / total;
  const over = (line: number) => matrix.reduce((sum, row, home) =>
    sum + row.reduce((rowSum, value, away) => rowSum + (home + away > line ? value : 0), 0), 0) / total;
  const bttsYes = probability((home, away) => home > 0 && away > 0);
  return { over, bttsYes, probability };
}

function fractionalOdds(probability: number) {
  if (probability <= 0) return "—";
  const decimal = (1 / probability) - 1;
  const approximate = (value: number): [number, number] => {
    let bestNumerator = 1;
    let bestDenominator = 1;
    let bestDifference = Number.POSITIVE_INFINITY;
    for (let denominator = 1; denominator <= 100; denominator += 1) {
      const numerator = Math.max(1, Math.round(value * denominator));
      const difference = Math.abs(value - numerator / denominator);
      if (difference < bestDifference) {
        bestNumerator = numerator;
        bestDenominator = denominator;
        bestDifference = difference;
      }
    }
    const gcd = (a: number, b: number): number => b ? gcd(b, a % b) : a;
    const divisor = gcd(bestNumerator, bestDenominator);
    return [bestNumerator / divisor, bestDenominator / divisor];
  };
  const [numerator, denominator] = approximate(decimal);
  return `${numerator}/${denominator}`;
}

function FairPrice({ probability, format, highlight = false }: { probability: number; format: OddsFormat; highlight?: boolean }) {
  const decimal = probability > 0 ? (1 / probability).toFixed(2) : "—";
  return <span className={`fair-price${highlight ? " fair-price-best" : ""}`} title="Model-implied fair price; not a bookmaker quote">{format === "decimal" ? decimal : fractionalOdds(probability)}</span>;
}

function TeamMark({ name, away = false, large = false }: { name: string; away?: boolean; large?: boolean }) {
  return <span className={`team-mark${away ? " team-mark-away" : ""}${large ? " team-mark-large" : ""}`} aria-hidden="true">{initials(name)}</span>;
}

function FixtureCard({ prediction, selected, format, onSelect }: { prediction: Prediction; selected: boolean; format: OddsFormat; onSelect: () => void }) {
  const outcomes = [prediction.home_win_probability, prediction.draw_probability, prediction.away_win_probability];
  const best = Math.max(...outcomes);
  const prices: [string, number][] = [["1", prediction.home_win_probability], ["X", prediction.draw_probability], ["2", prediction.away_win_probability]];
  return (
    <button type="button" className={`fixture-card${selected ? " fixture-card-selected" : ""}`} onClick={onSelect} aria-pressed={selected}>
      {selected && <span className="fixture-selected-rail" />}
      <div className="fixture-card-meta"><span>{formatDate(prediction.match_date)} <i>·</i> {prediction.kickoff_utc} UTC</span><span className="fixture-status"><b /> {prediction.evaluation_status === "pending" ? "Upcoming" : "Evaluated"}</span></div>
      <div className="fixture-teams">
        <span className="fixture-team"><TeamMark name={prediction.home_team} /> <span>{prediction.home_team}</span></span>
        <span className="fixture-team"><TeamMark name={prediction.away_team} away /> <span>{prediction.away_team}</span></span>
      </div>
      <div className="fixture-prices" aria-label="Model probabilities and fair prices">
        {prices.map(([label, value]) => (
          <span className="fixture-price" key={label}><span>{label}</span><strong>{percent(value)}</strong><FairPrice probability={value} format={format} highlight={value === best} /></span>
        ))}
      </div>
    </button>
  );
}

function MarketRow({ label, probability, format, best = false, showFairPrice = true }: { label: string; probability: number; format: OddsFormat; best?: boolean; showFairPrice?: boolean }) {
  return (
    <div className="market-row">
      <span className={best ? "market-row-label market-row-best" : "market-row-label"}>{label}</span>
      <span className="market-track"><span className={best ? "market-track-fill market-track-best" : "market-track-fill"} style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} /></span>
      <strong className="market-percent">{percent(probability)}</strong>
      {showFairPrice ? <FairPrice probability={probability} format={format} highlight={best} /> : <span className="fair-price">—</span>}
    </div>
  );
}

function DetailPanel({ prediction, format }: { prediction: Prediction; format: OddsFormat }) {
  const outcomes = [
    { key: "1", name: prediction.home_team, probability: prediction.home_win_probability },
    { key: "X", name: "Draw", probability: prediction.draw_probability },
    { key: "2", name: prediction.away_team, probability: prediction.away_win_probability },
  ];
  const best = Math.max(...outcomes.map((outcome) => outcome.probability));
  const topPick = outcomes.find((outcome) => outcome.probability === best)!;
  const markets = goalMarkets(prediction);
  const teamCoverage = [prediction.home_team_history_available, prediction.away_team_history_available];
  const missingTeams = [
    !prediction.home_team_history_available ? prediction.home_team : "",
    !prediction.away_team_history_available ? prediction.away_team : "",
  ].filter(Boolean);

  return (
    <div className="detail-stack">
      <section className="match-hero">
        <div className="detail-meta"><span>{prediction.competition ?? "Competition"} <i>·</i> {formatDate(prediction.match_date, { weekday: undefined, year: "numeric" })}</span><span>{prediction.kickoff_utc} UTC</span></div>
        <div className="hero-teams">
          <div className="hero-team"><TeamMark name={prediction.home_team} large /><strong>{prediction.home_team}</strong><small>HOME</small></div>
          <div className="hero-score"><span>EXPECTED GOALS</span><strong>{prediction.home_expected_goals.toFixed(2)} <i>–</i> {prediction.away_expected_goals.toFixed(2)}</strong><small>model estimate</small></div>
          <div className="hero-team"><TeamMark name={prediction.away_team} away large /><strong>{prediction.away_team}</strong><small>AWAY</small></div>
        </div>
        <div className="top-pick"><span>Most likely 1X2 outcome</span><strong>{topPick.key} <i>·</i> {topPick.name}</strong><span className="top-pick-prob">{percent(topPick.probability)}</span></div>
      </section>

      <section className="snapshot-grid" aria-label="Match data snapshot">
        <article className="snapshot-card"><span>MODEL</span><strong>Joint attack / defence</strong><small>Version {prediction.model_version ?? "0.1.0"}</small></article>
        <article className="snapshot-card"><span>TEAM HISTORY</span><strong>{teamCoverage.filter(Boolean).length} of 2 teams</strong><small>{missingTeams.length ? "League prior used where needed" : "Both teams have history"}</small></article>
        <article className="snapshot-card"><span>FORECAST SAVED</span><strong>{formatGenerated(prediction.prediction_timestamp)}</strong><small>Preserved before evaluation</small></article>
      </section>

      {missingTeams.length > 0 && <div className="coverage-note" role="note"><span className="note-icon">i</span><span>{missingTeams.join(" and ")} lack historical team records; league-prior estimates are used.</span></div>}

      <div className="markets-heading"><div><span className="section-kicker">PROBABILITY BREAKDOWN</span><h2>Match markets</h2></div><span className="odds-note">Model fair prices</span></div>
      <MarketTabs prediction={prediction} format={format} markets={markets} best={best} />
    </div>
  );
}

function MarketTabs({ prediction, format, markets, best }: { prediction: Prediction; format: OddsFormat; markets: ReturnType<typeof goalMarkets>; best: number }) {
  const [tab, setTab] = useState<MarketTab>("result");
  const outcomes = [
    { key: "1", label: `1 · ${prediction.home_team}`, p: prediction.home_win_probability },
    { key: "X", label: "X · Draw", p: prediction.draw_probability },
    { key: "2", label: `2 · ${prediction.away_team}`, p: prediction.away_win_probability },
  ];
  const tabItems: { id: MarketTab; label: string }[] = [
    { id: "result", label: "Match result" }, { id: "goals", label: "Goals" }, { id: "btts", label: "BTTS" },
    { id: "combos", label: "Combos" }, { id: "team-goals", label: "Team goals" },
    { id: "handicap", label: "Handicap" }, { id: "scores", label: "Likely scores" },
  ];
  const scoreItems: { score: string; probability?: number; rank: number }[] = [
    { score: prediction.score_1, probability: prediction.score_1_probability, rank: 1 },
    { score: prediction.score_2, probability: prediction.score_2_probability, rank: 2 },
    { score: prediction.score_3, probability: prediction.score_3_probability, rank: 3 },
  ];
  return (
    <section className="market-section">
      <div className="market-tabs" role="tablist" aria-label="Prediction markets">
        {tabItems.map((item) => <button type="button" role="tab" aria-selected={tab === item.id} className={tab === item.id ? "market-tab market-tab-active" : "market-tab"} onClick={() => setTab(item.id)} key={item.id}>{item.label}</button>)}
      </div>
      {tab === "result" && <div className="market-content">
        <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">FULL TIME</span><h3>1X2 result</h3></div><span>Probability <i>·</i> fair price</span></div>{outcomes.map((item) => <MarketRow key={item.key} label={item.label} probability={item.p} format={format} best={item.p === best} />)}</div>
        <div className="market-two-col">
          <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">COVERAGE</span><h3>Double chance</h3></div></div>
            <MarketRow label="1X · Home or draw" probability={prediction.home_win_probability + prediction.draw_probability} format={format} />
            <MarketRow label="X2 · Draw or away" probability={prediction.draw_probability + prediction.away_win_probability} format={format} />
            <MarketRow label="12 · Either team wins" probability={prediction.home_win_probability + prediction.away_win_probability} format={format} />
          </div>
          <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">DRAW REFUNDED</span><h3>Draw no bet</h3></div></div>
            <MarketRow label={prediction.home_team} probability={prediction.home_win_probability / (prediction.home_win_probability + prediction.away_win_probability)} format={format} />
            <MarketRow label={prediction.away_team} probability={prediction.away_win_probability / (prediction.home_win_probability + prediction.away_win_probability)} format={format} />
          </div>
        </div>
      </div>}
      {tab === "goals" && <div className="market-panel market-content"><div className="panel-heading"><div><span className="section-kicker">DERIVED FROM EXPECTED GOALS</span><h3>Over / under goals</h3></div><span>Model estimate</span></div>
        {[0.5, 1.5, 2.5, 3.5].map((line) => { const over = markets.over(line); return <div className="goal-line" key={line}><strong className="goal-line-name">{line} goals</strong><div className="goal-split"><div><span>Over</span><b>{percent(over)}</b><i><span style={{ width: `${over * 100}%` }} /></i></div><div><span>Under</span><b>{percent(1 - over)}</b><i><span style={{ width: `${(1 - over) * 100}%` }} /></i></div></div></div>; })}
      </div>}
      {tab === "btts" && <div className="market-panel market-content"><div className="panel-heading"><div><span className="section-kicker">BOTH TEAMS TO SCORE</span><h3>GG / NG</h3></div><span>Derived from expected goals</span></div><div className="btts-cards"><div className="btts-card"><span>GG <i>·</i> Yes</span><strong>{percent(markets.bttsYes)}</strong><FairPrice probability={markets.bttsYes} format={format} /></div><div className="btts-card"><span>NG <i>·</i> No</span><strong>{percent(1 - markets.bttsYes)}</strong><FairPrice probability={1 - markets.bttsYes} format={format} /></div></div><div className="btts-track"><span style={{ width: `${markets.bttsYes * 100}%` }} /></div></div>}
      {tab === "combos" && <div className="market-content market-two-col">
        <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">COMBINED FULL-TIME MARKET</span><h3>1X2 + Over/Under</h3></div><span>Joint score probabilities</span></div>
          {[1.5, 2.5].flatMap((line) => outcomes.flatMap((outcome, index) => {
            const outcomeTest = (home: number, away: number) => index === 0 ? home > away : index === 1 ? home === away : home < away;
            return [
              { label: `${outcome.key} + Over ${line}`, p: markets.probability((home, away) => outcomeTest(home, away) && home + away > line) },
              { label: `${outcome.key} + Under ${line}`, p: markets.probability((home, away) => outcomeTest(home, away) && home + away < line) },
            ];
          })).map((item) => <MarketRow key={item.label} label={item.label} probability={item.p} format={format} />)}
        </div>
        <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">COMBINED FULL-TIME MARKET</span><h3>Double Chance + GG/NG</h3></div><span>Joint score probabilities</span></div>
          {[
            { label: "1X + GG", condition: (home: number, away: number) => home >= away && home > 0 && away > 0 },
            { label: "1X + NG", condition: (home: number, away: number) => home >= away && (home === 0 || away === 0) },
            { label: "X2 + GG", condition: (home: number, away: number) => home <= away && home > 0 && away > 0 },
            { label: "X2 + NG", condition: (home: number, away: number) => home <= away && (home === 0 || away === 0) },
            { label: "12 + GG", condition: (home: number, away: number) => home !== away && home > 0 && away > 0 },
            { label: "12 + NG", condition: (home: number, away: number) => home !== away && (home === 0 || away === 0) },
          ].map((item) => <MarketRow key={item.label} label={item.label} probability={markets.probability(item.condition)} format={format} />)}
        </div>
        <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">FULL-TIME GOALS</span><h3>Odd / Even</h3></div><span>All goals</span></div>
          <MarketRow label="Odd total goals" probability={markets.probability((home, away) => (home + away) % 2 === 1)} format={format} />
          <MarketRow label="Even total goals" probability={markets.probability((home, away) => (home + away) % 2 === 0)} format={format} />
        </div>
        <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">FULL-TIME GOALS</span><h3>MultiGoal ranges</h3></div><span>All goals</span></div>
          {[
            { label: "0–1 goals", condition: (home: number, away: number) => home + away <= 1 },
            { label: "2–3 goals", condition: (home: number, away: number) => home + away >= 2 && home + away <= 3 },
            { label: "4+ goals", condition: (home: number, away: number) => home + away >= 4 },
          ].map((item) => <MarketRow key={item.label} label={item.label} probability={markets.probability(item.condition)} format={format} />)}
        </div>
      </div>}
      {tab === "team-goals" && <div className="market-content market-two-col">
        {[{ name: prediction.home_team, home: true }, { name: prediction.away_team, home: false }].map((team) => <div className="market-panel" key={team.name}>
          <div className="panel-heading"><div><span className="section-kicker">TEAM TOTAL GOALS</span><h3>{team.name}</h3></div><span>Full time</span></div>
          {[0.5, 1.5, 2.5].map((line) => {
            const over = markets.probability((home, away) => (team.home ? home : away) > line);
            return <div className="goal-line" key={line}><strong className="goal-line-name">{line}</strong><div className="goal-split"><div><span>Over</span><b>{percent(over)}</b></div><div><span>Under</span><b>{percent(1 - over)}</b></div></div></div>;
          })}
        </div>)}
      </div>}
      {tab === "handicap" && <div className="market-content market-two-col">
        <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">FULL-TIME HANDICAP</span><h3>European Handicap</h3></div><span>Home team line</span></div>
          {[-1, 1].map((line) => <Fragment key={line}>
            <div className="section-kicker" style={{ marginTop: 10 }}>HOME {line > 0 ? `+${line}` : line}</div>
            {[
              { label: "Home wins", condition: (home: number, away: number) => home + line > away },
              { label: "Draw", condition: (home: number, away: number) => home + line === away },
              { label: "Away wins", condition: (home: number, away: number) => home + line < away },
            ].map((item) => <MarketRow key={`${line}-${item.label}`} label={item.label} probability={markets.probability(item.condition)} format={format} />)}
          </Fragment>)}
        </div>
        <div className="market-panel"><div className="panel-heading"><div><span className="section-kicker">FULL-TIME HANDICAP</span><h3>Asian Handicap</h3></div><span>Whole-goal lines</span></div>
          {[-1, 0, 1].map((line) => <div key={line}>
            <div className="section-kicker" style={{ marginTop: 10 }}>HOME {line > 0 ? `+${line}` : line === 0 ? "0" : line}</div>
            {[
              { label: "Win", condition: (home: number, away: number) => home + line > away },
              { label: "Push · stake returned", condition: (home: number, away: number) => home + line === away },
              { label: "Lose", condition: (home: number, away: number) => home + line < away },
            ].map((item) => <MarketRow key={`${line}-${item.label}`} label={item.label} probability={markets.probability(item.condition)} format={format} showFairPrice={false} />)}
          </div>)}
          <p className="markets-disclaimer">Quarter-goal handicap lines can settle as half wins or half losses and are not included in this first pass.</p>
        </div>
      </div>}
      {tab === "scores" && <div className="market-panel market-content"><div className="panel-heading"><div><span className="section-kicker">SCORELINE PROBABILITIES</span><h3>Most likely scores</h3></div><span>From saved forecast</span></div><div className="score-grid">{scoreItems.map(({ score, probability, rank }) => <article className={`score-card${rank === 1 ? " score-card-best" : ""}`} key={score}><span>{rank === 1 ? "MOST LIKELY" : `RANK ${rank}`}</span><strong>{score}</strong><small>{percent(probability)}</small></article>)}</div></div>}
      <p className="markets-disclaimer">Probabilities are model estimates. Fair prices are calculated as 1 ÷ probability; they are not bookmaker odds or betting recommendations.</p>
    </section>
  );
}

export default function Home() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [selectedDate, setSelectedDate] = useState("all");
  const [competition, setCompetition] = useState("Premier League");
  const [oddsFormat, setOddsFormat] = useState<OddsFormat>("decimal");
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [predictionResponse, performanceResponse] = await Promise.all([
        fetch(`/api/football/predictions?${new URLSearchParams({ status: "pending", competition })}`, { cache: "no-store" }),
        fetch(`/api/football/performance?${new URLSearchParams({ competition })}`, { cache: "no-store" }),
      ]);
      if (!predictionResponse.ok || !performanceResponse.ok) throw new Error("API error");
      const [predictionData, performanceData] = await Promise.all([predictionResponse.json(), performanceResponse.json()]) as [PredictionResponse, Performance];
      setPredictions(predictionData.predictions);
      setPerformance(performanceData);
      setUpdatedAt(new Date().toISOString());
      setSelectedId((current) => predictionData.predictions.some((item) => item.prediction_id === current) ? current : predictionData.predictions[0]?.prediction_id ?? "");
    } catch {
      setError("Could not reach the prediction API. Start the Python service at 127.0.0.1:8000 and try again.");
    } finally { setLoading(false); }
  }, [competition]);

  useEffect(() => { void refresh(); }, [refresh]);
  const dates = useMemo(() => [...new Set(predictions.map((item) => item.match_date))].sort(), [predictions]);
  const filtered = useMemo(() => predictions.filter((item) => selectedDate === "all" || item.match_date === selectedDate), [predictions, selectedDate]);
  const selected = filtered.find((item) => item.prediction_id === selectedId) ?? filtered[0];
  useEffect(() => { if (selected && selected.prediction_id !== selectedId) setSelectedId(selected.prediction_id); }, [selected, selectedId]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#top" aria-label="Touchline home"><span className="brand-icon"><span /></span><span className="brand-name">touchline<span>.</span></span></a>
        <div className="nav-label">WORKSPACE</div>
        <nav className="main-nav" aria-label="Main navigation"><a className="nav-item active" href="#fixtures"><span className="nav-glyph">◫</span> Match centre</a><a className="nav-item" href="#performance"><span className="nav-glyph">⌁</span> Model performance</a></nav>
        <div className="sidebar-bottom"><div className="model-card"><span className="model-indicator" /><div><small>ACTIVE MODEL</small><strong>Joint Poisson</strong><span>Experimental</span></div></div><div className="sidebar-caption">Probabilities are estimates,<br />not guarantees.</div><div className="sidebar-version">ENGINE v0.1.0 <span>·</span> EPL</div></div>
      </aside>

      <section className="main-content" id="top">
        <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> Match centre</div><div className="topbar-right"><span className="live-dot" /> ENGINE CONNECTED <span className="topbar-divider" /> <span className="topbar-league">{competition.toUpperCase()}</span></div></header>
        <nav className="league-nav" aria-label="Competition filter">{LEAGUES.map((league) => <button type="button" key={league} className={competition === league ? "league-pill league-pill-active" : "league-pill"} aria-pressed={competition === league} onClick={() => { setCompetition(league); setSelectedDate("all"); }}>{league}</button>)}</nav>
        <div className="content-wrap">
          <section className="page-heading"><div><div className="eyebrow"><span /> MATCHDAY FORECASTS</div><h1>See the game<br /><em>in probabilities.</em></h1><p className="page-subtitle">Evidence-led forecasts for the fixtures ahead. Every number carries uncertainty.</p></div><div className="heading-actions"><div className="odds-toggle" role="group" aria-label="Fair price format"><button type="button" className={oddsFormat === "decimal" ? "toggle-active" : ""} onClick={() => setOddsFormat("decimal")}>Decimal</button><button type="button" className={oddsFormat === "fractional" ? "toggle-active" : ""} onClick={() => setOddsFormat("fractional")}>Fractional</button></div><button className="refresh-button" onClick={() => void refresh()} disabled={loading}><span className={loading ? "refresh-icon spinning" : "refresh-icon"}>↻</span> Refresh</button></div></section>

          <section className="summary-grid" id="performance" aria-label="Model summary"><div className="summary-card summary-primary"><div className="summary-label">UPCOMING FORECASTS <span>↗</span></div><div className="summary-value">{loading ? "—" : String(predictions.length).padStart(2, "0")}</div><div className="summary-foot">Saved before kickoff</div></div><div className="summary-card"><div className="summary-label">EVALUATED MATCHES</div><div className="summary-value">{performance?.evaluated_count ?? "—"}</div><div className="summary-foot">Held in forecast history</div></div><div className="summary-card"><div className="summary-label">OUTCOME ACCURACY</div><div className="summary-value">{percent(performance?.outcome_accuracy)}</div><div className="summary-foot">Historical, not a promise</div></div><div className="summary-card"><div className="summary-label">MULTICLASS BRIER</div><div className="summary-value">{performance?.multiclass_brier_score?.toFixed(3) ?? "—"}</div><div className="summary-foot">Lower is better</div></div></section>

          <section className="workspace-section" id="fixtures"><div className="section-heading"><div><div className="section-kicker">{competition.toUpperCase()} · THE NEXT ROUND</div><h2>Upcoming fixtures <span>{filtered.length.toString().padStart(2, "0")}</span></h2></div><div className="fixture-tools"><label className="date-filter">{dates.length > 1 ? "Match date" : "Fixture date"}<select value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)}><option value="all">All dates</option>{dates.map((date) => <option key={date} value={date}>{formatDate(date, { weekday: undefined, year: "numeric" })}</option>)}</select></label>{updatedAt && <div className="updated-label">UPDATED {formatGenerated(updatedAt).toUpperCase()}</div>}</div></div>
            {error && <div className="error-panel" role="alert"><strong>Connection unavailable</strong><span>{error}</span><button onClick={() => void refresh()}>Try again</button></div>}
            {loading && <div className="loading-panel"><span className="loading-spinner" /> Loading forecasts from the engine…</div>}
            {!loading && !error && predictions.length === 0 && <div className="empty-panel">There are no saved {competition} forecasts yet. Add sourced fixtures and generate forecasts for this competition to see them here.</div>}
            {!loading && !error && predictions.length > 0 && filtered.length === 0 && <div className="empty-panel">There are no saved {competition} forecasts yet. Add sourced fixtures and generate forecasts for this competition to see them here.</div>}
            {!loading && !error && filtered.length > 0 && <div className="fixture-layout"><aside className="fixture-feed"><div className="feed-header"><span>{competition.toUpperCase()}</span><strong>{filtered.length} {filtered.length === 1 ? "match" : "matches"}</strong></div>{filtered.map((prediction) => <FixtureCard key={prediction.prediction_id} prediction={prediction} selected={prediction.prediction_id === selected?.prediction_id} format={oddsFormat} onSelect={() => { setSelectedId(prediction.prediction_id); setMobileDetailOpen(true); }} />)}</aside><section className="detail-column" aria-label="Selected match breakdown">{selected ? <DetailPanel prediction={selected} format={oddsFormat} /> : <div className="empty-panel">Choose a fixture to see the forecast breakdown.</div>}</section></div>}
          </section>
          <footer className="page-footer"><span>TOUCHLINE <i>·</i> FOOTBALL PREDICTION ENGINE</span><span>Model outputs describe probability, never certainty.</span></footer>
        </div>
      </section>

      {mobileDetailOpen && selected && <div className="mobile-drawer-backdrop" role="presentation" onClick={() => setMobileDetailOpen(false)}><section className="mobile-drawer" role="dialog" aria-modal="true" aria-label={`${selected.home_team} vs ${selected.away_team} match breakdown`} onClick={(event) => event.stopPropagation()}><div className="drawer-handle" /><button type="button" className="drawer-close" onClick={() => setMobileDetailOpen(false)} aria-label="Close match breakdown">×</button><DetailPanel prediction={selected} format={oddsFormat} /></section></div>}
    </main>
  );
}
