import { useData, pct } from "../lib/useData.js";
import { ProbBar, SkeletonPage, ErrorState, Stagger, StaggerItem } from "../components/ui.jsx";

function GroupCard({ g }) {
  return (
    <StaggerItem className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display text-lg">Group {g.group}</h3>
        <span className="text-xs text-white/40">advance probability</span>
      </div>
      <div className="space-y-3">
        {g.teams.map((t, i) => (
          <div key={t.team}>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="flex items-center gap-2">
                <span className="w-5 text-white/30 tabular">{i + 1}</span>
                <span>{t.team}</span>
                {t.played > 0 && (
                  <span className="text-xs text-white/35 tabular">
                    {t.points}pt · {t.gd >= 0 ? "+" : ""}{t.gd}
                  </span>
                )}
              </span>
              <span className="tabular text-white/70">{pct(t.p_advance, 0)}</span>
            </div>
            <ProbBar
              p={t.p_advance}
              color={t.p_advance > 0.66 ? "var(--color-accent)" : t.p_advance > 0.33 ? "var(--color-warn)" : "#6b7a76"}
              height={6}
            />
          </div>
        ))}
      </div>
    </StaggerItem>
  );
}

export default function Groups() {
  const groups = useData("groups.json");
  if (groups.loading) return <SkeletonPage />;
  if (groups.error) return <ErrorState error={groups.error} />;

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <h1 className="text-3xl font-display font-bold mb-2">Group stage</h1>
      <p className="text-white/55 mb-8 max-w-2xl">
        Top two from each group advance, plus the eight best third-placed teams. Bars show each
        team's simulated probability of reaching the round of 32.
      </p>
      <Stagger className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
        {groups.data.groups.map((g) => <GroupCard key={g.group} g={g} />)}
      </Stagger>
    </div>
  );
}
