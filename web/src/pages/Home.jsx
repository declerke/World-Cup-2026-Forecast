import { Link } from "react-router-dom";
import { useData, pct } from "../lib/useData.js";
import { Stat, TriProbBar, SkeletonPage, ErrorState, Stagger, StaggerItem } from "../components/ui.jsx";

function UpcomingCard({ m }) {
  const p = m.probs || {};
  return (
    <Link to={`/match/${m.match_id}`}>
      <StaggerItem className="card lift p-4 h-full">
        <div className="flex items-center justify-between text-xs text-white/40 mb-3">
          <span>{m.stage === "group" ? `Group ${m.group}` : m.stage}</span>
          <span>{new Date(m.utc_date).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
        </div>
        <div className="flex items-center justify-between font-display">
          <span className="truncate">{m.home_team}</span>
          <span className="text-white/30 text-sm px-2">vs</span>
          <span className="truncate text-right">{m.away_team}</span>
        </div>
        {m.probs && (
          <div className="mt-3">
            <TriProbBar pHome={p.p_home} pDraw={p.p_draw} pAway={p.p_away} />
            <div className="flex justify-between text-xs mt-1.5 tabular">
              <span className="text-[var(--color-accent)]">{pct(p.p_home, 0)}</span>
              <span className="text-white/40">{pct(p.p_draw, 0)} draw</span>
              <span className="text-[var(--color-warn)]">{pct(p.p_away, 0)}</span>
            </div>
          </div>
        )}
      </StaggerItem>
    </Link>
  );
}

function ChampionRace({ teams }) {
  const top = teams.slice(0, 10);
  const max = top[0]?.p_champion || 1;
  return (
    <div className="card p-6">
      <h2 className="font-display text-lg mb-4">Who wins the cup?</h2>
      <div className="space-y-3">
        {top.map((t, i) => (
          <div key={t.team} className="flex items-center gap-3">
            <span className="w-6 text-white/30 text-sm tabular">{i + 1}</span>
            <span className="w-20 sm:w-28 lg:w-36 truncate text-sm">{t.team}</span>
            <div className="flex-1 h-6 rounded-lg bg-white/5 overflow-hidden relative">
              <div className="prob-fill h-full rounded-lg bg-gradient-to-r from-[var(--color-accent-dim)] to-[var(--color-accent)]"
                   style={{ width: `${(t.p_champion / max) * 100}%` }} />
            </div>
            <span className="w-16 text-right tabular text-sm font-medium">{pct(t.p_champion)}</span>
            {t.delta_vs_yesterday != null && t.delta_vs_yesterday !== 0 && (
              <span className={`hidden sm:block w-12 text-right text-xs tabular ${t.delta_vs_yesterday > 0 ? "text-[var(--color-accent)]" : "text-[var(--color-loss)]"}`}>
                {t.delta_vs_yesterday > 0 ? "▲" : "▼"}{Math.abs(t.delta_vs_yesterday * 100).toFixed(1)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const meta = useData("meta.json");
  const odds = useData("champion_odds.json");
  const matches = useData("matches.json");
  const acc = useData("accuracy.json");

  if (meta.loading || odds.loading || matches.loading) return <SkeletonPage />;
  if (odds.error || matches.error) return <ErrorState error={odds.error || matches.error} />;

  const teams = odds.data.teams;
  const favorite = teams[0];
  const mover = [...teams].filter((t) => t.delta_vs_yesterday != null)
    .sort((a, b) => Math.abs(b.delta_vs_yesterday) - Math.abs(a.delta_vs_yesterday))[0];
  const now = Date.now();
  const upcoming = matches.data.matches
    .filter((m) => m.status !== "FINISHED" && m.home_team && m.away_team && new Date(m.utc_date).getTime() > now - 6 * 3600e3)
    .sort((a, b) => new Date(a.utc_date) - new Date(b.utc_date))
    .slice(0, 6);
  const summary = acc.data?.summary;

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <section className="mb-10">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--color-accent)] mb-3">
          FIFA World Cup 2026 · Live ML Forecast
        </div>
        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-display font-bold leading-[1.05] max-w-3xl">
          Every match. Every probability.{" "}
          <span className="text-[var(--color-accent)]">Updated daily.</span>
        </h1>
        <p className="mt-4 text-white/55 max-w-2xl">
          A machine-learning model rates all 48 teams, predicts every remaining fixture, and
          simulates the tournament {meta.data ? meta.data.n_sims.toLocaleString() : "10,000"} times.
          Predictions are frozen before kickoff and graded in the open.
        </p>
      </section>

      <Stagger className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        <Stat label="Favorite" value={favorite.team} sub={`${pct(favorite.p_champion)} to win`} accent />
        <Stat label="Biggest mover" value={mover ? mover.team : "—"}
              sub={mover ? `${mover.delta_vs_yesterday > 0 ? "+" : ""}${(mover.delta_vs_yesterday * 100).toFixed(1)} pts` : "since yesterday"} />
        <Stat label="Model accuracy" value={summary?.favorite_accuracy != null ? pct(summary.favorite_accuracy) : "—"}
              sub={summary?.n_scored ? `${summary.n_scored} graded` : "tracking begins"} />
        <Stat label="Simulations" value={meta.data ? `${(meta.data.n_sims / 1000).toFixed(0)}k` : "10k"}
              sub="per daily run" />
      </Stagger>

      <div className="grid lg:grid-cols-2 gap-6">
        <ChampionRace teams={teams} />
        <div>
          <h2 className="font-display text-lg mb-4">Next up</h2>
          <Stagger className="grid sm:grid-cols-2 gap-3">
            {upcoming.map((m) => <UpcomingCard key={m.match_id} m={m} />)}
          </Stagger>
          {upcoming.length === 0 && <p className="text-white/40 text-sm">No upcoming fixtures.</p>}
        </div>
      </div>
    </div>
  );
}
