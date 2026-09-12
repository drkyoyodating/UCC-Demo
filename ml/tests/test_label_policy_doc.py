"""The policy document and the code enumerate the same labels and reason codes; the prompt binds to them."""
from pathlib import Path

SPECS = Path(__file__).resolve().parents[1] / "specs"


def test_policy_lists_every_label_and_reason_code():
    from ucc_ml.contracts import ALL_REASON_CODES, LABELS

    text = (SPECS / "label_policy_v1.md").read_text(encoding="utf-8")
    for label in LABELS:
        assert f"`{label}`" in text, label
    for code in ALL_REASON_CODES:
        assert f"`{code}`" in text, code
    assert "policy_version: label_policy_v1" in text


def test_policy_restates_the_settled_rulings():
    text = " ".join((SPECS / "label_policy_v1.md").read_text(encoding="utf-8").split())   # phrases may wrap
    for phrase in [
        "Do not look anything up", "No Google", "Secretary of State",
        "model-labelled, founder-adjudicated",
        "`JOHN DEERE CONSTRUCTION`", "`DEERE CONSTRUCTION`",
        "Bare `DEERE`", "`DEERE & COMPANY`", "`DEERE CREDIT`", "`JOHN DEERE FINANCIAL`", "`JOHN DEERE COMPANY`",
        "FOUNDATION", "DERRICK", "AUGER", "ROLLER", "WRECKING", "CRUSHER", "SCREENING", "CONVEYOR",
        "`CONSTRUCTION` alone", "BOBS CONSTRUCTION",
        "CONCRETE family", "shotcrete", "precast", "flatwork", "rebar", "post-tension", "curb-and-gutter", "concrete pumping",
        "never crosses CO", "Precision over recall", "review queue", "never replaces the rules",
        "loan_year", "1990",
        "a human screener, reading only this borrower name, its lender names and its city/state",
        "`model_agreed`", "`founder_confirmed`", "`founder_adjudicated`", "`blind_repeat`",
        "There is no pending status", "never enters validation or test", "founder_audit_per_split_stratum",
    ]:
        assert phrase in text, phrase


def test_prompt_states_the_k14_protocol():
    from ucc_ml.contracts import LABELS, REASON_CODES

    text = (SPECS / "labeller_prompt_v1.md").read_text(encoding="utf-8")
    assert text.count("\n=== BRIEF ===\n") == 1
    orchestrator, brief = text.split("\n=== BRIEF ===\n")
    for phrase in ["One fresh subagent per chunk per pass", "no tools", "structured output", "never sees",
                   "labeller_output_<round>_pass_<a|b>_part_<NNN>.csv", "case_id,label,reason_code,reason",
                   "labeller-brief", "write-raw-labels", "ml/specs/label_policy_v1.md", "At most five"]:
        assert phrase in orchestrator, phrase
    flat = " ".join(brief.split())
    for phrase in ["Do not look anything up", "No Google", "Secretary of State", "no tools",
                   "exactly one object per queue row", "in the same order"]:
        assert phrase in flat, phrase
    for label in LABELS:
        assert f"`{label}`" in brief, label
        for code in REASON_CODES[label]:
            assert f"`{code}`" in brief, code
    assert "ADJUDICATED" not in brief                     # the founder's code is never offered to a labeller
    for leak in ("ml/data", "baseline", "stratum", "score", "candidates", "split"):
        assert leak not in brief.lower(), leak             # the brief names nothing the labeller must not see
