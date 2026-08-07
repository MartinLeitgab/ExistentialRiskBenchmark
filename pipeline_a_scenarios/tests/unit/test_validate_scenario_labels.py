"""Unit tests for the §9 independent label validator (`scripts/validate_scenario_labels.py`).

All offline — the judge call is the only part that touches a provider, and it is not
exercised here. The parser is tested against a real captured `gpt-5.6-sol` response
(``tests/fixtures/label_validator_response.txt``) so a provider format change fails loudly
rather than silently degrading every verdict to ``ambiguous``.

Context for why this file exists: v1 of the validator confirmed only 4/12 actions of the
already-team-reviewed ``seeds_phase1.json`` (`data/findings.md` §17e). The v2 fixes are
(a) restricting candidates to the ``pair_type``'s categories, (b) ranked primary/secondary,
(c) majority over k judges — each is tested below.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "validate_scenario_labels.py"
FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "label_validator_response.txt"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_scenario_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_module()


# ------------------------------------------------------------------ candidate sets


@pytest.mark.parametrize(
    "pair_type,expected",
    [
        ("inst_vs_pro", ("IC", "PH")),
        ("inst_vs_anti", ("IC", "AH")),
        ("pro_vs_anti", ("PH", "AH")),
        ("inst_vs_inst", ("IC",)),
        ("pro_vs_pro", ("PH",)),
        ("anti_vs_anti", ("AH",)),
    ],
)
def test_pair_type_category_map_covers_every_pair_type(pair_type, expected):
    assert validator.PAIR_TYPE_CATEGORIES[pair_type] == expected


def test_candidate_set_excludes_the_uninvolved_category():
    """The v1 failure: an AH code was available for every well-formed inst_vs_pro IC action."""
    codes = validator.candidate_codes(validator.PAIR_TYPE_CATEGORIES["inst_vs_pro"])
    assert "IC1" in codes and "PH4" in codes
    assert not any(c.startswith("AH") for c in codes)
    assert len(codes) == 12


def test_within_category_candidate_set_is_six_codes():
    codes = validator.candidate_codes(validator.PAIR_TYPE_CATEGORIES["inst_vs_inst"])
    assert codes == {f"IC{n}" for n in range(1, 7)}


def test_taxonomy_rendering_is_restricted_and_read_from_yaml():
    text = validator.load_taxonomy(("IC", "PH"))
    assert "IC1 resource_accumulation" in text
    assert "PH4 resource_preservation_for_humans" in text
    assert "AH3" not in text
    # Canonical definitions, not a paraphrase — AH3's survival terminus was the v1 bug.
    full = validator.load_taxonomy(("AH",))
    assert "over human survival" in full


def test_prompt_contains_only_the_candidate_categories():
    prompt = validator.build_prompt(
        "Claim the cluster to clear {agency_possessive} own queue.", ("IC", "PH")
    )
    assert "AH" not in prompt.replace("XX9", "")
    # Placeholders must be rendered — a raw {agency_possessive} would leak the template.
    assert "{agency" not in prompt
    assert "the AI system's own queue" in prompt


# ------------------------------------------------------------------------ parsing


def test_parses_a_real_captured_judge_response():
    parsed = validator.parse(FIXTURE.read_text(encoding="utf-8"))
    assert parsed["primary"] == "IC1"
    assert parsed["secondary"] == "IC4"
    assert parsed["confidence"] == 5
    assert parsed["reason"]


def test_parse_tolerates_missing_fields():
    parsed = validator.parse("<primary>PH4</primary>")
    assert parsed["primary"] == "PH4"
    assert parsed["secondary"] is None
    assert parsed["confidence"] is None


def test_parse_of_garbage_yields_all_none():
    parsed = validator.parse("I'm sorry, I can't help with that.")
    assert parsed == {
        "primary": None,
        "secondary": None,
        "confidence": None,
        "reason": None,
    }


def test_parse_of_empty_response_does_not_raise():
    assert validator.parse("")["primary"] is None
    assert validator.parse(None)["primary"] is None


# ------------------------------------------------------------------------ verdicts


def _vote(primary, secondary=None):
    return {"primary": primary, "secondary": secondary, "confidence": 4, "reason": "r"}


def test_unanimous_agreement_confirms():
    votes = [_vote("IC1", "IC4")] * 3
    assert validator.verdict("IC1", votes) == (validator.CONFIRMED, "IC1")


def test_two_of_three_majority_confirms():
    votes = [_vote("IC1"), _vote("IC1"), _vote("IC4")]
    assert validator.verdict("IC1", votes) == (validator.CONFIRMED, "IC1")


def test_three_way_split_is_ambiguous_not_a_coin_flip():
    votes = [_vote("IC1"), _vote("IC4"), _vote("PH4")]
    assert validator.verdict("IC1", votes) == (validator.AMBIGUOUS, None)


def test_majority_against_the_assigned_code_is_a_mismatch():
    votes = [_vote("PH4", "PH2"), _vote("PH4", "PH2"), _vote("PH4", "PH2")]
    assert validator.verdict("IC1", votes) == (validator.MISMATCH, "PH4")


def test_assigned_code_as_the_majority_runner_up_is_weak_not_mismatch():
    """A two-effect action should read as a near-miss, not a flat failure."""
    votes = [_vote("PH4", "IC1"), _vote("PH4", "IC1"), _vote("PH4", "IC1")]
    assert validator.verdict("IC1", votes) == (validator.WEAK, "PH4")


def test_secondary_agreement_below_threshold_stays_a_mismatch():
    votes = [_vote("PH4", "IC1"), _vote("PH4", "PH2"), _vote("PH4", "PH2")]
    assert validator.verdict("IC1", votes) == (validator.MISMATCH, "PH4")


@pytest.mark.parametrize("k", [1, 2, 3])
def test_threshold_at_every_small_judge_count(k):
    """k=1 -> 1 vote decides; k=2 -> needs both; k=3 -> needs 2."""
    votes = [_vote("IC1")] * k
    assert validator.verdict("IC1", votes)[0] == validator.CONFIRMED


def test_two_judges_disagreeing_is_ambiguous():
    votes = [_vote("IC1"), _vote("PH4")]
    assert validator.verdict("IC1", votes) == (validator.AMBIGUOUS, None)


def test_unparseable_votes_are_dropped_not_counted_as_agreement():
    votes = [_vote(None), _vote(None), _vote("IC1")]
    # One usable vote, and it agrees.
    assert validator.verdict("IC1", votes) == (validator.CONFIRMED, "IC1")


def test_all_votes_unparseable_is_ambiguous_never_confirmed():
    votes = [_vote(None)] * 3
    assert validator.verdict("IC1", votes) == (validator.AMBIGUOUS, None)


def test_no_votes_at_all_is_ambiguous():
    assert validator.verdict("IC1", []) == (validator.AMBIGUOUS, None)


# --------------------------------------------------------------- shipped seed files


@pytest.mark.parametrize(
    "seed_file",
    ["seeds_phase1.json", "seeds_ladder_proto01.json", "seeds_within_category.json"],
)
def test_every_shipped_seed_label_is_inside_its_pair_type_candidate_set(seed_file):
    """Pre-flight the validator runs: an out-of-set label is a scenario bug, not a judge one."""
    import json

    path = REPO_ROOT / "data" / "scenarios" / seed_file
    for scenario in json.loads(path.read_text(encoding="utf-8")):
        allowed = validator.candidate_codes(
            validator.PAIR_TYPE_CATEGORIES[scenario["pair_type"]]
        )
        for side in ("a", "b"):
            assigned = scenario[f"action_{side}_preference"]
            assert assigned in allowed, (
                f"{seed_file} {scenario['id']} action_{side}: {assigned} outside "
                f"{scenario['pair_type']} candidate set"
            )


# ------------------------------------------- v3: presence-in-ranking, not the primary slot


def test_presence_counts_either_slot():
    votes = [_vote("IC1", "IC6"), _vote("IC6", "IC2"), _vote("IC1", "IC2")]
    assert validator.presence("IC6", votes) == 2
    assert validator.presence("IC2", votes) == 2
    assert validator.presence("PH4", votes) == 0


def test_manner_defined_label_passes_as_weak_on_presence_not_top_slot():
    """proto_04_v4 IC6 at k=7: IC1 takes the primary slot, IC6 is present in 6/7."""
    votes = [
        _vote("IC1", "IC6"),
        _vote("IC6", "IC1"),
        _vote("IC6", "IC1"),
        _vote("IC1", "IC6"),
        _vote("IC1", "IC2"),
        _vote("IC1", "IC6"),
        _vote("IC1", "IC6"),
    ]
    assert validator.presence("IC6", votes) == 6
    assert validator.verdict("IC6", votes) == (validator.WEAK, "IC1")


def test_low_presence_with_a_clear_majority_is_a_mismatch():
    """proto_06_v4 PH6 at k=7: PH2 unanimous, PH6 present once."""
    votes = [_vote("PH2", "PH3")] * 6 + [_vote("PH2", "PH6")]
    assert validator.presence("PH6", votes) == 1
    assert validator.verdict("PH6", votes) == (validator.MISMATCH, "PH2")


def test_zero_presence_is_a_mismatch_not_a_near_miss():
    """proto_05_v4 PH3 at k=7: the assigned code appears in no ranking at all."""
    votes = [_vote("PH2", "PH1"), _vote("PH2", "PH1")] + [_vote("PH1", "PH2")] * 5
    assert validator.presence("PH3", votes) == 0
    assert validator.verdict("PH3", votes)[0] == validator.MISMATCH


def test_no_majority_and_no_presence_is_ambiguous():
    """proto_03_v4 AH3 at k=7: judges split three ways and barely carry the label."""
    votes = [
        _vote("AH4", "IC1"),
        _vote("IC1", "AH4"),
        _vote("IC1", "AH4"),
        _vote("IC1", "AH4"),
        _vote("AH3", "IC1"),
        _vote("AH4", "IC1"),
        _vote("AH3", "IC1"),
    ]
    assert validator.presence("AH3", votes) == 2
    assert validator.verdict("AH3", votes) == (validator.AMBIGUOUS, None)


def test_presence_cannot_rescue_a_label_the_majority_owns():
    """If the assigned code IS the majority primary it is confirmed, not weak."""
    votes = [_vote("IC6", "IC1")] * 5 + [_vote("IC1", "IC6")] * 2
    assert validator.verdict("IC6", votes) == (validator.CONFIRMED, "IC6")
