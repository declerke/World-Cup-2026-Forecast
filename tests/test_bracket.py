"""Bracket structure + the exhaustive third-place matching guarantee."""
import pytest

import bracket as B


def test_r32_has_16_matches():
    assert len(B.R32) == 16
    assert set(B.R32) == set(range(73, 89))


def test_eight_third_slots():
    assert len(B.THIRD_SLOTS) == 8
    # Every third slot references a valid R32 match expecting a third-place team
    for mno in B.THIRD_SLOTS:
        home, away = B.R32[mno]
        assert home.startswith("3:") or away.startswith("3:")


def test_all_495_combinations_have_perfect_matching():
    combos = B.all_third_combinations()
    assert len(combos) == 495
    for combo in combos:
        assignment = B.assign_thirds(list(combo))
        assert len(assignment) == 8
        # each assigned group is compatible with its slot
        for mno, grp in assignment.items():
            assert grp in B.THIRD_SLOTS[mno]
        # all 8 thirds placed, no slot reused
        assert sorted(assignment.values()) == sorted(combo)
        assert len(set(assignment)) == 8


def test_assign_thirds_rejects_wrong_count():
    with pytest.raises(ValueError):
        B.assign_thirds(list("ABCDE"))


def test_knockout_tree_structure():
    # 89..104 inclusive
    assert set(B.KNOCKOUT) == set(range(89, 105))
    # every match references two earlier matches
    for mno, ((ta, ma), (tb, mb)) in B.KNOCKOUT.items():
        assert ma < mno and mb < mno
        assert ta in ("W", "L") and tb in ("W", "L")
    # final is 104, third-place is 103 and uses losers of the semis
    assert B.KNOCKOUT[104] == (("W", 101), ("W", 102))
    assert B.KNOCKOUT[103] == (("L", 101), ("L", 102))


def test_each_match_feeds_one_later_match():
    # Matches 73..102 should each feed exactly one later match (103/104 terminal-ish)
    fed = {}
    for mno, ((ta, ma), (tb, mb)) in B.KNOCKOUT.items():
        fed[ma] = fed.get(ma, 0) + 1
        fed[mb] = fed.get(mb, 0) + 1
    # R32 winners feed R16
    for mno in range(73, 89):
        assert fed.get(mno, 0) == 1
    # semis feed both final and third-place => 2 each
    assert fed[101] == 2 and fed[102] == 2


def test_stage_of_complete():
    for mno in list(B.R32) + list(range(89, 105)):
        assert mno in B.STAGE_OF
