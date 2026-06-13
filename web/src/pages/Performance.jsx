import { useData, pct } from "../lib/useData.js";
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip,
  ScatterChart, Scatter, ZAxis, ReferenceLine,
} from "recharts";
import { Stat, SkeletonPage, ErrorState, Stagger } from "../components/ui.jsx";

function Calibration({ bins }) {
  if (!bins || bins.length === 0)
    return <p className="text-white/40 text-sm">Not enough graded matches yet.</p>;
  const data = bins.map((b) => ({ x: b.predicted, y: b.actual, n: b.count }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
        <XAxis type="number" dataKey="x" domain={[0.3, 1]} name="predicted"
               tick={{ fontSize: 11, fill: "#6b7a76" }} tickFormatter={(v) => pct(v, 0)} />
        <YAxis type="number" dataKey="y" domain={[0, 1]} name="actual"
               tick={{ fontSize: 11, fill: "#6b7a76" }} tickFormatter={(v) => pct(v, 0)} />
        <ZAxis dataKey="n" range={[60, 400]} />
        <ReferenceLine segment={[{ x: 0.3, y: 0.3 }, { x: 1, y: 1 }]} stroke="#3a4a45" strokeDasharray="4 4" />
        <Tooltip contentStyle={{ background: "#151c1a", border: "1px solid #1f2926", borderRadius: 8 }}
                 formatter={(v) => pct(v)} />
        <Scatter data={data} fill="var(--color-accent)" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export default function Performance() {
  const acc = useData("accuracy.json");
  const meta = useData("meta.json");
  if (acc.loading || meta.loading) return <SkeletonPage />;
  if (acc.error) return <ErrorState error={acc.error} />;

  const s = acc.data.summary;
  const hasData = s.n_scored > 0;
  const m = meta.data?.metrics || {};

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <h1 className="text-3xl font-display font-bold mb-2">How good is the model?</h1>
      <p className="text-white/55 mb-8 max-w-2xl">
        Every prediction is frozen before kickoff and graded once the result is known. Nothing here
        is fitted after the fact — these are honest, out-of-sample receipts.
      </p>

      <Stagger className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        <Stat label="Matches graded" value={hasData ? s.n_scored : "0"} sub="frozen pre-kickoff" />
        <Stat label="Favorite hit rate" value={hasData ? pct(s.favorite_accuracy) : "—"} accent
              sub="model's top pick won" />
        <Stat label="Mean Brier" value={hasData ? s.mean_brier.toFixed(3) : "—"} sub="lower is better" />
        <Stat label="Exact scores" value={hasData ? pct(s.exact_accuracy) : "—"} sub="hardest to call" />
      </Stagger>

      <div className="grid md:grid-cols-2 gap-6 mb-10">
        <div className="card p-5">
          <h3 className="font-display mb-1">Calibration</h3>
          <p className="text-xs text-white/45 mb-4">
            Dots on the diagonal mean the model's confidence matches reality.
          </p>
          <Calibration bins={acc.data.calibration} />
        </div>
        <div className="card p-5">
          <h3 className="font-display mb-1">Accuracy over time</h3>
          <p className="text-xs text-white/45 mb-4">Cumulative favorite hit-rate as matches are graded.</p>
          {hasData && acc.data.history.length > 1 ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={acc.data.history}>
                <XAxis dataKey="match" tick={{ fontSize: 11, fill: "#6b7a76" }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#6b7a76" }} tickFormatter={(v) => pct(v, 0)} />
                <Tooltip contentStyle={{ background: "#151c1a", border: "1px solid #1f2926", borderRadius: 8 }} />
                <Line type="monotone" dataKey="cum_favorite_accuracy" stroke="var(--color-accent)" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-white/40 text-sm py-20 text-center">First matchday pending.</p>
          )}
        </div>
      </div>

      <div className="card p-6 mb-10">
        <h3 className="font-display mb-4">Model card</h3>
        <div className="grid sm:grid-cols-2 gap-x-8 gap-y-2 text-sm text-white/60">
          <div className="flex justify-between"><span>Held-out test log loss</span><span className="tabular text-white">{m.test_logloss?.toFixed(4) ?? "—"}</span></div>
          <div className="flex justify-between"><span>Beats Elo-only baseline</span><span className="text-[var(--color-accent)]">{m.beats_elo_only ? "yes" : "—"}</span></div>
          <div className="flex justify-between"><span>Held-out accuracy</span><span className="tabular text-white">{m.test_accuracy ? pct(m.test_accuracy) : "—"}</span></div>
          <div className="flex justify-between"><span>Beats base-rate baseline</span><span className="text-[var(--color-accent)]">{m.beats_base_rate ? "yes" : "—"}</span></div>
          <div className="flex justify-between"><span>Training matches</span><span className="tabular text-white">{meta.data?.training_rows?.toLocaleString() ?? "—"}</span></div>
          <div className="flex justify-between"><span>Simulations / run</span><span className="tabular text-white">{meta.data?.n_sims?.toLocaleString() ?? "—"}</span></div>
        </div>
        <p className="text-xs text-white/45 mt-5 leading-relaxed">
          The model uses Elo ratings, recent form, strength of schedule, head-to-head history and
          match importance. It does <span className="text-white/70">not</span> see injuries, suspensions,
          lineups, weather or in-tournament morale. International match prediction tops out around
          55–60% accuracy for anyone — bookmakers included — because football is low-scoring and
          high-variance. Treat every number as a probability, not a promise.
        </p>
      </div>

      {hasData && (
        <div className="card p-6">
          <h3 className="font-display mb-4">Receipts — what we said, what happened</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-white/40 text-left text-xs">
                  <th className="py-2 pr-4">Match</th>
                  <th className="py-2 pr-4">Predicted</th>
                  <th className="py-2 pr-4">Result</th>
                  <th className="py-2">Call</th>
                </tr>
              </thead>
              <tbody>
                {acc.data.receipts.slice().reverse().map((r) => (
                  <tr key={r.match_id} className="border-t border-[var(--color-line)]">
                    <td className="py-2 pr-4">{r.home} v {r.away}</td>
                    <td className="py-2 pr-4 tabular text-white/60">{r.pred_score_home}–{r.pred_score_away}</td>
                    <td className="py-2 pr-4 tabular">{r.actual_home}–{r.actual_away}</td>
                    <td className="py-2">
                      <span className={Number(r.favorite_correct) ? "text-[var(--color-accent)]" : "text-[var(--color-loss)]"}>
                        {Number(r.favorite_correct) ? "✓" : "✗"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
