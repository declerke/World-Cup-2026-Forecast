"""2026 World Cup knockout bracket structure + third-place allocation.

All constants verified against Wikipedia "2026 FIFA World Cup knockout stage"
on 2026-06-13. The round of 32 introduces the eight best third-placed teams;
their slot assignment depends on which groups they come from (FIFA publishes a
495-row lookup table). Rather than hard-code 495 rows, we solve the assignment
as a constrained bipartite matching and verify a perfect matching exists for
every one of the C(12,8)=495 combinations in the test suite.
"""
from __future__ import annotations

from itertools import combinations

# ---- Round of 32: the 16 matches (verified) -------------------------------
# Each entry: match_no -> (home_slot, away_slot)
# Slots: "1X"/"2X" = winner/runner-up of group X; "3:SET" = a best-third team
#        whose group is in the allowed SET.
R32 = {
    73: ("2A", "2B"),
    74: ("1E", "3:ABCDF"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3:CDFGH"),
    78: ("2E", "2I"),
    79: ("1A", "3:CEFHI"),
    80: ("1L", "3:EHIJK"),
    81: ("1D", "3:BEFIJ"),
    82: ("1G", "3:AEHIJ"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3:EFGIJ"),
    86: ("1J", "2H"),
    87: ("1K", "3:DEIJL"),
    88: ("2D", "2G"),
}

# The eight R32 slots that receive a best-third team, with their allowed groups.
THIRD_SLOTS = {
    74: set("ABCDF"),
    77: set("CDFGH"),
    79: set("CEFHI"),
    80: set("EHIJK"),
    81: set("BEFIJ"),
    82: set("AEHIJ"),
    85: set("EFGIJ"),
    87: set("DEIJL"),
}

# ---- Round of 16 .. Final: winner-of-match feeds (verified) ----------------
# match_no -> (source_a, source_b); source = ("W", match) or ("L", match)
KNOCKOUT = {
    89: (("W", 74), ("W", 77)),
    90: (("W", 73), ("W", 75)),
    91: (("W", 76), ("W", 78)),
    92: (("W", 79), ("W", 80)),
    93: (("W", 83), ("W", 84)),
    94: (("W", 81), ("W", 82)),
    95: (("W", 86), ("W", 88)),
    96: (("W", 85), ("W", 87)),
    97: (("W", 89), ("W", 90)),
    98: (("W", 93), ("W", 94)),
    99: (("W", 91), ("W", 92)),
    100: (("W", 95), ("W", 96)),
    101: (("W", 97), ("W", 98)),
    102: (("W", 99), ("W", 100)),
    103: (("L", 101), ("L", 102)),   # third-place playoff
    104: (("W", 101), ("W", 102)),   # final
}

STAGE_OF = (
    {n: "R32" for n in R32}
    | {n: "R16" for n in range(89, 97)}
    | {n: "QF" for n in range(97, 101)}
    | {n: "SF" for n in (101, 102)}
    | {103: "3RD", 104: "FINAL"}
)


def assign_thirds(qualified_groups: list[str]) -> dict[int, str]:
    """Assign 8 best-third groups to the 8 third slots via bipartite matching.

    `qualified_groups` is the list of group letters (len 8) whose third-placed
    team advanced, in ranking order. Returns {match_no: group_letter}.
    Raises ValueError if no perfect matching exists (would indicate a constant
    error vs the official table).
    """
    if len(qualified_groups) != 8:
        raise ValueError(f"need exactly 8 third groups, got {len(qualified_groups)}")
    slots = list(THIRD_SLOTS.items())  # [(match_no, allowed_set), ...]
    assignment: dict[int, str] = {}
    used_slots: set[int] = set()

    def backtrack(i: int) -> bool:
        if i == len(qualified_groups):
            return True
        grp = qualified_groups[i]
        # prefer lowest-numbered available compatible slot (deterministic)
        for match_no, allowed in sorted(slots):
            if match_no in used_slots:
                continue
            if grp in allowed:
                used_slots.add(match_no)
                assignment[match_no] = grp
                if backtrack(i + 1):
                    return True
                used_slots.discard(match_no)
                del assignment[match_no]
        return False

    if not backtrack(0):
        raise ValueError(f"no perfect matching for third groups {qualified_groups}")
    return assignment


def all_third_combinations() -> list[tuple]:
    """All 495 ways to choose 8 of the 12 groups (for exhaustive testing)."""
    return list(combinations("ABCDEFGHIJKL", 8))


if __name__ == "__main__":
    ok = 0
    for combo in all_third_combinations():
        try:
            assign_thirds(list(combo))
            ok += 1
        except ValueError:
            print("FAILED:", combo)
    print(f"perfect matching for {ok}/495 third-place combinations")
