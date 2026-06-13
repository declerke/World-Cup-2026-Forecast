import { useState } from "react";
import { useData, pct } from "../lib/useData.js";
import { SkeletonPage, ErrorState } from "../components/ui.jsx";

const ROUNDS = [
  { key: "R32", label: "Round of 32", matches: range(73, 89) },
  { key: "R16", label: "Round of 16", matches: range(89, 97) },
  { key: "QF", label: "Quarter-finals", matches: range(97, 101) },
  { key: "SF", label: "Semi-finals", matches: [101, 102] },
  { key: "FINAL", label: "Final", matches: [104] },
];

function range(a, b) {
  return Array.from({ length: b - a }, (_, i) => a + i);
}

function SlotCard({ mno, slot, onPick, active }) {
  const top = slot?.top || [];
  const leader = top[0];
  return (
    <button
      onClick={() => onPick(active ? null : mno)}
      className={`card lift w-full text-left p-3 ${active ? "accent-glow" : ""}`}
    >
      <div className="text-[10px] uppercase tracking-wider text-white/30 mb-1">Match {mno}</div>
      {leader ? (
        <>
          <div className="flex items-center justify-between">
            <span className="text-sm truncate">{leader.team}</span>
            <span className="text-xs tabular text-[var(--color-accent)]">{pct(leader.p, 0)}</span>
          </div>
          {top[1] && (
            <div className="flex items-center justify-between text-white/40 mt-0.5">
              <span className="text-xs truncate">{top[1].team}</span>
              <span className="text-[10px] tabular">{pct(top[1].p, 0)}</span>
            </div>
          )}
        </>
      ) : (
        <div className="text-sm text-white/30">TBD</div>
      )}
      {active && top.length > 2 && (
        <div className="mt-2 pt-2 border-t border-[var(--color-line)] space-y-1">
          {top.slice(2, 6).map((t) => (
            <div key={t.team} className="flex justify-between text-xs text-white/50">
              <span className="truncate">{t.team}</span>
              <span className="tabular">{pct(t.p, 0)}</span>
            </div>
          ))}
        </div>
      )}
    </button>
  );
}

export default function Bracket() {
  const bracket = useData("bracket.json");
  const [active, setActive] = useState(null);
  if (bracket.loading) return <SkeletonPage />;
  if (bracket.error) return <ErrorState error={bracket.error} />;
  const slots = bracket.data.slots;

  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      <h1 className="text-3xl font-display font-bold mb-2">Knockout bracket</h1>
      <p className="text-white/55 mb-8 max-w-2xl">
        Most likely occupant of each bracket slot across {bracket.data ? "all" : ""} simulations.
        Tap a match to see the full distribution of teams that could fill it.
      </p>

      {/* Horizontal scroll on desktop, vertical accordion on mobile */}
      <div className="hidden md:grid gap-4" style={{ gridTemplateColumns: `repeat(${ROUNDS.length}, minmax(0,1fr))` }}>
        {ROUNDS.map((r) => (
          <div key={r.key}>
            <h2 className="text-sm font-display text-white/50 mb-3">{r.label}</h2>
            <div className="space-y-3">
              {r.matches.map((mno) => (
                <SlotCard key={mno} mno={mno} slot={slots[mno]} onPick={setActive} active={active === mno} />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="md:hidden space-y-6">
        {ROUNDS.map((r) => (
          <div key={r.key}>
            <h2 className="text-sm font-display text-white/50 mb-3">{r.label}</h2>
            <div className="grid grid-cols-2 gap-3">
              {r.matches.map((mno) => (
                <SlotCard key={mno} mno={mno} slot={slots[mno]} onPick={setActive} active={active === mno} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
