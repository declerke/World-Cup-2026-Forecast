"""Monte Carlo tournament simulation.

For each of N simulations we play out every remaining match, build group tables
with FIFA tiebreakers, rank the eight best third-placed teams, resolve the R32
bracket (including the constrained third-place allocation), then play the
knockout tree to a champion. Completed matches always use their real results.

Group-stage scorelines are pre-sampled from each fixture's renormalised Poisson
matrix, whose win/draw/loss mass already equals the classifier's — so sampled
outcomes are classifier-consistent by construction.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

import bracket as B
import config as C
import knockout as KO
import predict
import team_names as tn
import venues


class Simulator:
    def __init__(self, art: dict, fixtures, n_sims: int = C.N_SIMULATIONS, seed=None):
        self.states = art["states"]
        self.h2h = art["h2h"]
        self.final_elo = art["final_elo"]
        self.baselines = art.get("altitude_baselines", {})
        self.n = n_sims
        self.rng = np.random.default_rng(seed if seed is not None else C.run_seed())

        self.teams = tn.ALL_TEAMS
        self.tidx = {t: i for i, t in enumerate(self.teams)}
        self.elo = np.array([self.final_elo.get(t, C.ELO_START) for t in self.teams])
        self.group_of = {self.tidx[t]: g for t, g in tn.TEAM_GROUP.items()}
        self.group_teams = {g: [self.tidx[t] for t in ts] for g, ts in tn.GROUPS.items()}

        self.fixtures = fixtures
        self._prepare_group_fixtures()
        self._pair_cache: dict[tuple, tuple] = {}

    # -- setup ---------------------------------------------------------------
    def _prepare_group_fixtures(self):
        gf = self.fixtures[self.fixtures["stage"] == "group"].copy()
        self.group_matches = defaultdict(list)
        n = C.MAX_GOALS + 1
        for _, m in gf.iterrows():
            if m["home_team"] is None or m["away_team"] is None:
                continue
            h, a = self.tidx[m["home_team"]], self.tidx[m["away_team"]]
            g = m["group"]
            if m["played"] and m["home_score"] is not None:
                hg = np.full(self.n, int(m["home_score"]), dtype=np.int16)
                ag = np.full(self.n, int(m["away_score"]), dtype=np.int16)
            else:
                neutral = predict.is_neutral(m["home_team"])
                v_alt = venues.wc2026_altitude(m["home_team"], m["away_team"])
                mat = predict.scoreline_matrix(
                    m["home_team"], m["away_team"], neutral=neutral,
                    states=self.states, h2h=self.h2h, importance=4,
                    baselines=self.baselines, venue_altitude=v_alt)
                cells = self.rng.choice(mat.size, size=self.n, p=mat.ravel())
                hg = (cells // n).astype(np.int16)
                ag = (cells % n).astype(np.int16)
            self.group_matches[g].append((h, a, hg, ag))

    def _pair(self, a_idx: int, b_idx: int):
        """Cached neutral-venue (p_a_advance via sampling inputs)."""
        key = (a_idx, b_idx)
        if key not in self._pair_cache:
            a, b = self.teams[a_idx], self.teams[b_idx]
            p_a, p_d, p_b = predict.wdl_neutral(a, b, states=self.states, h2h=self.h2h)
            self._pair_cache[key] = (p_a, p_d, p_b)
        return self._pair_cache[key]

    # -- group stage ---------------------------------------------------------
    @staticmethod
    def _break_ties(tied, results, rng):
        pts = {t: 0 for t in tied}
        gd = {t: 0 for t in tied}
        s = set(tied)
        for h, a, hg, ag in results:
            if h in s and a in s:
                gd[h] += hg - ag
                gd[a] += ag - hg
                if hg > ag:
                    pts[h] += 3
                elif hg < ag:
                    pts[a] += 3
                else:
                    pts[h] += 1
                    pts[a] += 1
        jitter = {t: rng.random() for t in tied}
        return sorted(tied, key=lambda t: (-pts[t], -gd[t], -jitter[t]))

    def _rank_group(self, team_ids, results):
        pts = {t: 0 for t in team_ids}
        gd = {t: 0 for t in team_ids}
        gf = {t: 0 for t in team_ids}
        for h, a, hg, ag in results:
            gf[h] += hg
            gf[a] += ag
            gd[h] += hg - ag
            gd[a] += ag - hg
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1

        def primary(t):
            return (pts[t], gd[t], gf[t])

        order = sorted(team_ids, key=lambda t: (-pts[t], -gd[t], -gf[t]))
        final = []
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and primary(order[j]) == primary(order[i]):
                j += 1
            tied = order[i:j]
            final.extend(self._break_ties(tied, results, self.rng) if len(tied) > 1 else tied)
            i = j
        return final, pts, gd, gf

    # -- one simulation ------------------------------------------------------
    def _simulate_one(self, sim_i: int, counters: dict):
        winners_by_group = {}
        runners_by_group = {}
        third_candidates = []  # (pts, gd, gf, rand, group, team_idx)

        for g, matches in self.group_matches.items():
            results = [(h, a, int(hg[sim_i]), int(ag[sim_i])) for h, a, hg, ag in matches]
            order, pts, gd, gf = self._rank_group(self.group_teams[g], results)
            winners_by_group[g] = order[0]
            runners_by_group[g] = order[1]
            third = order[2]
            third_candidates.append((pts[third], gd[third], gf[third],
                                     self.rng.random(), g, third))
            for pos, t in enumerate(order):
                counters["group_finish"][t][pos] += 1

        # best 8 thirds
        third_candidates.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3]))
        best8 = third_candidates[:8]
        third_team_by_group = {g: t for *_, g, t in best8}
        qualified_groups = [g for *_, g, _t in best8]   # ranking order
        third_slot_group = B.assign_thirds(qualified_groups)

        def slot_team(token: str):
            if token.startswith("3:"):
                return None  # handled per-match below
            pos, grp = token[0], token[1]
            return winners_by_group[grp] if pos == "1" else runners_by_group[grp]

        winners = {}
        losers = {}
        # Round of 32
        for mno, (hs, as_) in B.R32.items():
            a = slot_team(hs) if not hs.startswith("3:") else third_team_by_group[third_slot_group[mno]]
            b = slot_team(as_) if not as_.startswith("3:") else third_team_by_group[third_slot_group[mno]]
            counters["reach"]["R32"][a] += 1
            counters["reach"]["R32"][b] += 1
            counters["slot"][mno][a] += 1
            counters["slot"][mno][b] += 1
            w, l = self._play(a, b)
            winners[mno], losers[mno] = w, l

        # Rounds 89..104
        for mno in range(89, 105):
            (ta, ma), (tb, mb) = B.KNOCKOUT[mno]
            a = winners[ma] if ta == "W" else losers[ma]
            b = winners[mb] if tb == "W" else losers[mb]
            stage = B.STAGE_OF[mno]
            if stage in counters["reach"]:        # "3RD" playoff is not a reach stage
                counters["reach"][stage][a] += 1
                counters["reach"][stage][b] += 1
            counters["slot"][mno][a] += 1
            counters["slot"][mno][b] += 1
            w, l = self._play(a, b)
            winners[mno], losers[mno] = w, l

        counters["reach"]["CHAMPION"][winners[104]] += 1
        counters["reach"]["THIRD"][winners[103]] += 1

    def _play(self, a: int, b: int):
        p_a, p_d, p_b = self._pair(a, b)
        win = KO.sample_winner(self.rng, p_a, p_d, p_b, self.elo[a], self.elo[b])
        return (a, b) if win == 0 else (b, a)

    # -- run + aggregate -----------------------------------------------------
    def run(self) -> dict:
        counters = {
            "reach": {s: Counter() for s in ("R32", "R16", "QF", "SF", "FINAL", "CHAMPION", "THIRD")},
            "group_finish": defaultdict(lambda: Counter()),
            "slot": defaultdict(lambda: Counter()),
        }
        for i in range(self.n):
            self._simulate_one(i, counters)
        return self._aggregate(counters)

    def _aggregate(self, counters: dict) -> dict:
        N = self.n
        teams_out = {}
        for t, i in self.tidx.items():
            gf = counters["group_finish"][i]
            teams_out[t] = {
                "team": t,
                "group": tn.TEAM_GROUP[t],
                "elo": round(float(self.elo[i]), 1),
                "p_win_group": round(gf[0] / N, 4),
                "p_runner_up": round(gf[1] / N, 4),
                "p_third": round(gf[2] / N, 4),
                "p_fourth": round(gf[3] / N, 4),
                "p_r32": round(counters["reach"]["R32"][i] / N, 4),
                "p_r16": round(counters["reach"]["R16"][i] / N, 4),
                "p_qf": round(counters["reach"]["QF"][i] / N, 4),
                "p_sf": round(counters["reach"]["SF"][i] / N, 4),
                "p_final": round(counters["reach"]["FINAL"][i] / N, 4),
                "p_champion": round(counters["reach"]["CHAMPION"][i] / N, 4),
            }
        # bracket slot distributions (top occupants per match)
        slots = {}
        for mno, cnt in counters["slot"].items():
            top = [{"team": self.teams[t], "p": round(c / N, 4)}
                   for t, c in cnt.most_common(6)]
            slots[mno] = {"stage": B.STAGE_OF[mno], "top": top}
        return {"teams": teams_out, "slots": slots, "n_sims": N}


if __name__ == "__main__":
    import sys
    import artifacts
    import ingest
    art = artifacts.load()
    fixtures = ingest.load_fixtures()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    sim = Simulator(art, fixtures, n_sims=n, seed=C.SEED)
    out = sim.run()
    ranked = sorted(out["teams"].values(), key=lambda d: -d["p_champion"])[:12]
    print(f"\nChampion odds (top 12, {n} sims):")
    for d in ranked:
        print(f"  {d['team']:<16} {d['p_champion']*100:5.1f}%   "
              f"(SF {d['p_sf']*100:4.1f}%  R16 {d['p_r16']*100:4.1f}%)")
