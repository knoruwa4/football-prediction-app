"use client";

import { useCallback, useEffect, useState } from "react";

type Prediction = {
  prediction_id: string;
  prediction_timestamp: string;
  match_date: string;
  kickoff_utc: string;
  home_team: string;
  away_team: string;
  home_team_history_available: boolean;
  away_team_history_available: boolean;
  model: string;
  home_win_probability: number;
  draw_probability: number;
  away_win_probability: number;
  home_expected_goals: number;
  away_expected_goals: number;
  score_1: string;
  score_2: string;
  score_3: string;
  evaluation_status: "pending" | "evaluated";
};

type PredictionResponse = { count: number; predictions: Prediction[] };
type Performance = {
  evaluated_count: number;
  pending_count: number;
  outcome_accuracy: number | null;
  multiclass_brier_score: number | null;
};

const percent = (value: number | null) =>
  value === null ? "—" : `${(value * 100).toFixed(1)}%`;

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${value}T12:00:00Z`));
}

function formatGenerated(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(new Date(value));
}

function shortTeam(name: string) {
  return name
    .replace("AFC ", "")
    .replace(" & Hove Albion", "")
    .replace(" United", " Utd");
}

function MatchCard({ prediction }: { prediction: Prediction }) {
  const markets = [
    { label: "1", value: prediction.home_win_probability },
    { label: "X", value: prediction.draw_probability },
    { label: "2", value: prediction.away_win_probability },
  ];
  const homeKnown = prediction.home_team_history_available;
  const awayKnown = prediction.away_team_history_available;

  return (
    <article className="match-card">
      <div className="match-topline">
        <span className="competition"><span className="competition-mark">PL</span> Premier League</span>
        <span className="match-time">{prediction.kickoff_utc} UTC</span>
      </div>
      <div className="teams-row">
        <div className="team team-home">
          <span className="team-badge">{shortTeam(prediction.home_team).slice(0, 2).toUpperCase()}</span>
          <span>{prediction.home_team}</span>
        </div>
        <span className="versus">VS</span>
        <div className="team team-away">
          <span>{prediction.away_team}</span>
          <span className="team-badge away-badge">{shortTeam(prediction.away_team).slice(0, 2).toUpperCase()}</span>
        </div>
      </div>
      <div className="market-heading">
        <span>Match result · 1X2</span>
        <span className="model-tag">JOINT POISSON</span>
      </div>
      <div className="probability-row" aria-label="1X2 probabilities">
        {markets.map((market) => (
          <div className="probability" key={market.label}>
            <div className="probability-label"><span>{market.label}</span><strong>{percent(market.value)}</strong></div>
            <div className="probability-track"><span style={{ width: `${market.value * 100}%` }} /></div>
          </div>
        ))}
      </div>
      <div className="match-details">
        <div className="xg-detail"><span>Expected goals</span><strong>{prediction.home_expected_goals.toFixed(2)} <i>—</i> {prediction.away_expected_goals.toFixed(2)}</strong></div>
        <div className="scores-detail"><span>Likeliest scores</span><strong>{prediction.score_1}<i>·</i>{prediction.score_2}<i>·</i>{prediction.score_3}</strong></div>
      </div>
      {(!homeKnown || !awayKnown) && (
        <div className="coverage-note" role="note">
          <span className="note-icon">!</span>
          {[
            !homeKnown && prediction.home_team,
            !awayKnown && prediction.away_team,
          ].filter(Boolean).join(" and ")} lack historical team records; league-prior estimates are used.
        </div>
      )}
      <div className="card-footer">
        <span className="pending-label"><span /> Awaiting result</span>
        <span>Generated {formatGenerated(prediction.prediction_timestamp)}</span>
      </div>
    </article>
  );
}

export default function Home() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [predictionResponse, performanceResponse] = await Promise.all([
        fetch("/api/football/predictions?status=pending", { cache: "no-store" }),
        fetch("/api/football/performance", { cache: "no-store" }),
      ]);
      if (!predictionResponse.ok || !performanceResponse.ok) {
        throw new Error("The prediction service returned an error.");
      }
      const predictionData = (await predictionResponse.json()) as PredictionResponse;
      const performanceData = (await performanceResponse.json()) as Performance;
      setPredictions(predictionData.predictions);
      setPerformance(performanceData);
      setUpdatedAt(new Date().toISOString());
    } catch {
      setError("Could not reach the prediction API. Make sure the Python API is running at 127.0.0.1:8000.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#top" aria-label="Touchline home">
          <span className="brand-icon"><span /></span>
          <span className="brand-name">touchline<span>.</span></span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav className="main-nav" aria-label="Main navigation">
          <a className="nav-item active" href="#fixtures"><span className="nav-glyph">◫</span> Match centre</a>
          <a className="nav-item" href="#performance"><span className="nav-glyph">⌁</span> Model performance</a>
        </nav>
        <div className="sidebar-bottom">
          <div className="model-card">
            <span className="model-indicator" />
            <div><small>ACTIVE MODEL</small><strong>Joint Poisson</strong><span>Experimental</span></div>
          </div>
          <div className="sidebar-caption">Probabilities are estimates,<br />not guarantees.</div>
          <div className="sidebar-version">ENGINE v0.1.0 <span>·</span> EPL</div>
        </div>
      </aside>

      <section className="main-content" id="top">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> Match centre</div>
          <div className="topbar-right"><span className="live-dot" /> ENGINE CONNECTED <span className="topbar-divider" /> <span className="topbar-league">ENGLISH PREMIER LEAGUE</span></div>
        </header>

        <div className="content-wrap" id="fixtures">
          <section className="page-heading">
            <div>
              <div className="eyebrow"><span /> MATCHDAY FORECASTS</div>
              <h1>See the game<br /><em>in probabilities.</em></h1>
              <p className="page-subtitle">Evidence-led forecasts for the fixtures ahead. Every number carries uncertainty.</p>
            </div>
            <button className="refresh-button" onClick={() => void refresh()} disabled={loading}>
              <span className={loading ? "refresh-icon spinning" : "refresh-icon"}>↻</span> Refresh data
            </button>
          </section>

          <section className="summary-grid" id="performance" aria-label="Model summary">
            <div className="summary-card summary-primary"><div className="summary-label">UPCOMING FORECASTS <span>↗</span></div><div className="summary-value">{loading ? "—" : String(predictions.length).padStart(2, "0")}</div><div className="summary-foot">Saved before kickoff</div></div>
            <div className="summary-card"><div className="summary-label">EVALUATED MATCHES</div><div className="summary-value">{performance?.evaluated_count ?? "—"}</div><div className="summary-foot">Held in forecast history</div></div>
            <div className="summary-card"><div className="summary-label">OUTCOME ACCURACY</div><div className="summary-value">{percent(performance?.outcome_accuracy ?? null)}</div><div className="summary-foot">Historical, not a promise</div></div>
            <div className="summary-card"><div className="summary-label">MULTICLASS BRIER</div><div className="summary-value">{performance?.multiclass_brier_score?.toFixed(3) ?? "—"}</div><div className="summary-foot">Lower is better</div></div>
          </section>

          <section className="fixtures-section">
            <div className="section-heading">
              <div><div className="section-kicker">THE NEXT ROUND</div><h2>Upcoming fixtures <span>{predictions.length.toString().padStart(2, "0")}</span></h2></div>
              {updatedAt && <div className="updated-label">UPDATED {formatGenerated(updatedAt).toUpperCase()}</div>}
            </div>
            {error && <div className="error-panel" role="alert"><strong>Connection unavailable</strong><span>{error}</span><button onClick={() => void refresh()}>Try again</button></div>}
            {loading && <div className="loading-panel"><span className="loading-spinner" /> Loading forecasts from the engine…</div>}
            {!loading && !error && predictions.length === 0 && <div className="empty-panel">No saved upcoming forecasts yet. Once fixtures are verified and predicted, they will appear here.</div>}
            {!loading && !error && predictions.length > 0 && (
              <>
                <div className="date-divider"><span>{formatDate(predictions[0].match_date)}</span><i /></div>
                <div className="match-grid">{predictions.map((prediction) => <MatchCard key={prediction.prediction_id} prediction={prediction} />)}</div>
              </>
            )}
          </section>

          <footer className="page-footer"><span>TOUCHLINE <i>·</i> FOOTBALL PREDICTION ENGINE</span><span>Model outputs describe probability, never certainty.</span></footer>
        </div>
      </section>
    </main>
  );
}
