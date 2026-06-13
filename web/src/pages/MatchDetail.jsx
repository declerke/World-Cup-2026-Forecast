import { useParams, Link } from "react-router-dom";
import { useData, pct } from "../lib/useData.js";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { SkeletonPage, ErrorState, TriProbBar } from "../components/ui.jsx";

function FormStrip({ results }) {
  const color = { W: "var(--color-accent)", D: "#6b7a76", L: "var(--color-loss)" };
  return (
    <div className="flex gap-1">
      {results.length === 0 && <span className="text-white/30 text-xs">no data</span>}
      {results.map((r, i) => (
        <span key={i} className="w-6 h-6 rounded grid place-items-center text-xs font-medium"
              style={{ background: `${color[r]}22`, color: color[r] }}>{r}</span>
      ))}
    </div>
  );
}

function ShapPanel({ shap, home, away }) {
  const maxAbs = Math.max(...shap.map((s) => Math.abs(s.impact)), 0.0001);
  return (
    <div className="card p-5">
      <h3 className="font-display mb-1">Why the model leans this way</h3>
      <p className="text-xs text-white/45 mb-4">
        Top factors pushing toward the favored outcome (SHAP values).
      </p>
      <div className="space-y-2.5">
        {shap.map((s, i) => (
          <div key={i} className="flex items-center gap-3 text-sm">
            <span className="w-44 truncate text-white/70" title={s.feature}>{s.feature}</span>
            <div className="flex-1 h-4 relative bg-white/5 rounded">
              <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/15" />
              <div className="prob-fill absolute top-0 bottom-0 rounded"
                   style={{
                     width: `${(Math.abs(s.impact) / maxAbs) * 50}%`,
                     left: s.impact >= 0 ? "50%" : undefined,
                     right: s.impact < 0 ? "50%" : undefined,
                     background: s.impact >= 0 ? "var(--color-accent)" : "var(--color-loss)",
                   }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function MatchDetail() {
  const { id } = useParams();
  const d = useData(`match_detail/${id}.json`);
  if (d.loading) return <SkeletonPage />;
  if (d.error) return <ErrorState error={d.error} />;
  const m = d.data;
  const p = m.probs;
  const fav = Math.max(p.p_home, p.p_draw, p.p_away);
  const favLabel = fav === p.p_home ? m.home_team : fav === p.p_away ? m.away_team : "Draw";

  const eloData = (m.elo_history.home || []).map((h, i) => ({
    date: h.date, home: h.elo, away: m.elo_history.away[i]?.elo,
  }));

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <Link to="/" className="text-sm text-white/50 hover:text-white">← back</Link>
      <div className="text-xs uppercase tracking-wider text-[var(--color-accent)] mt-4 mb-2">
        {m.stage === "group" ? `Group ${m.group}` : m.stage} ·{" "}
        {new Date(m.utc_date).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
      </div>
      <h1 className="text-3xl sm:text-4xl font-display font-bold mb-6">
        {m.home_team} <span className="text-white/30">vs</span> {m.away_team}
      </h1>

      <div className="card p-6 mb-6">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-[var(--color-accent)] font-medium">{m.home_team} {pct(p.p_home)}</span>
          <span className="text-white/50">Draw {pct(p.p_draw)}</span>
          <span className="text-[var(--color-warn)] font-medium">{pct(p.p_away)} {m.away_team}</span>
        </div>
        <TriProbBar pHome={p.p_home} pDraw={p.p_draw} pAway={p.p_away} />
        <div className="mt-4 text-sm text-white/60">
          Model favors <span className="text-white font-medium">{favLabel}</span>. Most likely score:{" "}
          <span className="text-white tabular">{m.top_scorelines[0].home}–{m.top_scorelines[0].away}</span>
        </div>
      </div>

      {m.venue && (
        <div className="card p-5 mb-6" style={{ borderColor: "rgba(255,179,0,0.3)", background: "linear-gradient(rgba(255,179,0,0.05), rgba(255,179,0,0.02)), linear-gradient(180deg, #151c1a, #111715)" }}>
          <div className="flex items-start gap-3">
            <span className="text-2xl mt-0.5">⛰️</span>
            <div className="text-sm">
              <div className="font-display text-white mb-1">
                Altitude factor — {m.venue.city} ({m.venue.altitude_m.toLocaleString()} m)
              </div>
              <p className="text-white/60">
                {m.venue.favours
                  ? (() => {
                      const disadvantaged = m.venue.favours === m.home_team ? m.away_team : m.home_team;
                      const ascent = m.venue.favours === m.home_team ? m.venue.ascent_away_m : m.venue.ascent_home_m;
                      return <>
                        This match is played at altitude.{" "}
                        <span className="text-white">{m.venue.favours}</span> is the more acclimatised side.{" "}
                        <span className="text-white">{disadvantaged}</span> climbs{" "}
                        {ascent.toLocaleString()} m above their usual playing elevation, a documented physical disadvantage the model accounts for.
                      </>;
                    })()
                  : <>Both teams ascend a similar amount to reach this venue, so altitude is roughly neutral here.</>}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-6">
        <ShapPanel shap={m.shap} home={m.home_team} away={m.away_team} />

        <div className="card p-5">
          <h3 className="font-display mb-4">Likely scorelines</h3>
          <div className="space-y-2">
            {m.top_scorelines.map((s, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="tabular w-12 text-center font-display">{s.home}–{s.away}</span>
                <div className="flex-1 h-4 bg-white/5 rounded overflow-hidden">
                  <div className="prob-fill h-full bg-[var(--color-accent-dim)]"
                       style={{ width: `${(s.prob / m.top_scorelines[0].prob) * 100}%` }} />
                </div>
                <span className="tabular w-12 text-right text-sm text-white/60">{pct(s.prob)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card p-5">
          <h3 className="font-display mb-4">Recent form (last 10)</h3>
          <div className="space-y-3">
            <div>
              <div className="text-sm text-white/60 mb-1">{m.home_team}</div>
              <FormStrip results={m.form.home} />
            </div>
            <div>
              <div className="text-sm text-white/60 mb-1">{m.away_team}</div>
              <FormStrip results={m.form.away} />
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-[var(--color-line)] text-sm text-white/60">
            Head-to-head (last {m.h2h.meetings}): {m.home_team} {m.h2h.home_wins}–
            {m.h2h.draws}–{m.h2h.away_wins} {m.away_team}
          </div>
        </div>

        <div className="card p-5">
          <h3 className="font-display mb-4">Elo trend (2 yrs)</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={eloData}>
              <XAxis dataKey="date" hide />
              <YAxis domain={["auto", "auto"]} width={36} tick={{ fontSize: 11, fill: "#6b7a76" }} />
              <Tooltip contentStyle={{ background: "#151c1a", border: "1px solid #1f2926", borderRadius: 8 }} />
              <Line type="monotone" dataKey="home" stroke="var(--color-accent)" dot={false} strokeWidth={2} name={m.home_team} />
              <Line type="monotone" dataKey="away" stroke="var(--color-warn)" dot={false} strokeWidth={2} name={m.away_team} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
