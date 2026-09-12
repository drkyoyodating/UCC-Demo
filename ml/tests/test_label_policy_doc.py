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


# --- policy v2: the de-circularised brief, and the provenance guarantees it rests on ---

V1_SHA256 = "e6ca4e3975bab3c22a4df075ace549ed9a47df1f0ec3729366acecfff3114678"
#: Rosters v1 handed the labeller. v2 must not name any of them: naming one supplies the answer key
#: for the exact instrument the ablation is trying to measure.
WITHDRAWN = ("CATERPILLAR", "KOMATSU", "WAGNER EQUIPMENT", "TEREX", "LIEBHERR", "TAKEUCHI", "BOBCAT",
             "VOLVO", "HITACHI", "DITCH WITCH", "MANITOWOC", "DOOSAN", "VERMEER", "MANUFACTURERS",
             "TELEHANDLER", "BULLDOZER", "SKIDSTEER", "EARTHMOVING")
#: Brands that MUST survive: each teaches what does NOT qualify, which is a founder ruling, not a roster.
OVERRIDES = ("JOHN DEERE CONSTRUCTION", "DEERE CONSTRUCTION", "DEERE & COMPANY", "DEERE CREDIT",
             "JOHN DEERE FINANCIAL", "KUBOTA", "1ST SOURCE BANK")


def test_policy_v1_is_never_edited():
    """v1 is the 240 committed pilot rows' policy and the ablation's control arm. Editing it destroys
    the instrument and silently re-writes the provenance of labels already published."""
    import hashlib

    assert hashlib.sha256((SPECS / "label_policy_v1.md").read_bytes()).hexdigest() == V1_SHA256


def test_policy_v2_withdraws_every_roster_but_keeps_every_ruling():
    text = (SPECS / "label_policy_v2.md").read_text(encoding="utf-8")
    upper = text.upper()
    assert "policy_version: label_policy_v2" in text
    assert any(line.strip() == "status: FROZEN" for line in text.splitlines())
    for absent in WITHDRAWN:
        assert absent not in upper, f"v2 still names {absent}"
    for present in OVERRIDES:
        assert present in upper, f"v2 lost the ruling that names {present}"
    for phrase in ("FOUNDATION", "DERRICK", "AUGER", "ROLLER", "WRECKING", "CRUSHER", "SCREENING",
                   "CONVEYOR", "`CONSTRUCTION` alone", "shotcrete", "precast", "flatwork",
                   "post-tension", "curb-and-gutter", "Do not look anything up",
                   "model-labelled, founder-adjudicated", "never crosses CO ↔ CT"):
        assert phrase in text, f"v2 lost: {phrase}"
    # the two founder rulings of 2026-09-12
    assert "Each filing is judged on its own evidence" in text
    assert "Ordinary world knowledge is allowed" in text
    # v2 must not ask the labeller for a field the structured-output schema rejects
    from ucc_ml.labeling import RAW_OUTPUT_COLUMNS

    assert "lender_recognised" not in text
    assert set(RAW_OUTPUT_COLUMNS) == {"case_id", "label", "reason_code", "reason"}


def test_policy_version_is_resolved_per_round_not_globally():
    """Stamping one global version would put label_policy_v2 on the 240 rows the v1 brief produced."""
    from ucc_ml.config import load_config
    from ucc_ml.labeling import policy_version_for_round

    cfg = load_config(SPECS.parent / "configs" / "v1.yaml")
    assert policy_version_for_round(cfg, "pilot_v1") == "label_policy_v1"
    for later in ("ablation_v1", "yield_probe_v1", "main_v1", "queue_v1"):
        assert policy_version_for_round(cfg, later) == "label_policy_v2"


def test_ablation_round_has_its_own_files_and_never_collides_with_the_main_round():
    """round_paths branched only on pilot_v1, so every other round shared the main round's files."""
    from ucc_ml.config import load_config
    from ucc_ml.labeling import round_paths

    cfg = load_config(SPECS.parent / "configs" / "v1.yaml")
    ablation, main, pilot = (round_paths(cfg, r) for r in ("ablation_v1", "main_v1", "pilot_v1"))
    assert ablation.cases != main.cases != pilot.cases
    assert ablation.cases.name == "ablation_cases.parquet"
    assert ablation.report != main.report
    assert ablation.key != main.key and ablation.preregistration != main.preregistration
