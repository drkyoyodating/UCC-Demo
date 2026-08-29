"""Tests for src/normalize.py.

Every fixture case is driven from tests/fixtures/messy_names.csv, whose expected
values were hand-derived from the written rules by three independent passes
BEFORE src/normalize.py existed and agreed unanimously on all 50 rows.  The
fixture is therefore an external check on the implementation, not a transcript
of it -- if you change the normalisation rules, change the fixture by hand and
re-derive, do not regenerate it from the function.

The sys.path bootstrap below exists because this project has no conftest.py and
no packaging metadata: pytest puts tests/ on sys.path, not the repository root,
so `import src.normalize` would otherwise fail.  src/ has no __init__.py and is
imported as an implicit namespace package.
"""

import csv
import doctest
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import normalize as normalize_module  # noqa: E402
from src.normalize import COLUMN_MAP, normalize_name  # noqa: E402

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "messy_names.csv"

#: The 18 spellings the brief requires the normaliser to recognise.
BRIEF_SUFFIX_SPELLINGS = [
    "LLC", "L.L.C.", "INC", "INCORPORATED", "CORP", "CORPORATION",
    "CO", "COMPANY", "LTD", "LIMITED", "LP", "L.P.", "LLP", "PLLC",
    "PC", "P.C.", "NA", "N.A.",
]


def _load_fixture() -> list[tuple[str, str | None, str | None]]:
    """Read the fixture, skipping the leading '#' comment block.

    An empty expected field means Python None; that mapping is unambiguous
    because normalize_name never returns "" (asserted in
    test_never_returns_empty_string).
    """
    with FIXTURE.open(encoding="utf-8", newline="") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    rows = []
    for row in csv.DictReader(lines):
        rows.append((
            row["raw"],
            row["expected_name_clean"] or None,
            row["expected_suffix"] or None,
        ))
    return rows


FIXTURE_ROWS = _load_fixture()


# --------------------------------------------------------------------------
# The fixture
# --------------------------------------------------------------------------

def test_fixture_has_expected_case_count():
    # 50 original rows (categories C1-C8) + 6 added with step 5 (category C9).
    assert len(FIXTURE_ROWS) == 56


def test_fixture_covers_every_suffix_spelling_from_the_brief():
    """Each of the 18 spellings must appear as a trailing token somewhere."""
    raws = [r[0].upper().rstrip() for r in FIXTURE_ROWS]
    missing = [
        s for s in BRIEF_SUFFIX_SPELLINGS
        if not any(r == s or (r.endswith(s) and not r[: -len(s)][-1:].isalnum()) for r in raws)
    ]
    assert missing == [], f"fixture never exercises trailing {missing}"


@pytest.mark.parametrize(
    ("raw", "expected_clean", "expected_suffix"),
    FIXTURE_ROWS,
    ids=[f"row{i:02d}" for i in range(1, len(FIXTURE_ROWS) + 1)],
)
def test_fixture_case(raw, expected_clean, expected_suffix):
    assert normalize_name(raw) == (expected_clean, expected_suffix)


# --------------------------------------------------------------------------
# Null-ish and hostile inputs -- the single most likely crash site
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", " ", "\t", "\n", "\t \n ", "...", "!!!", "---", "'", "()",
     "©", "�", "中文"],
    ids=["none", "empty", "3spaces", "1space", "tab", "newline", "mixed_ws",
         "dots", "bangs", "dashes", "apostrophe", "parens",
         "copyright", "replacement_char", "cjk_only"],
)
def test_nullish_and_punctuation_only_input(raw):
    assert normalize_name(raw) == (None, None)


def test_bare_ampersand_becomes_the_word_and():
    """'&' -> ' AND ' is a word substitution, not punctuation stripping.

    So a name that is only '&' normalises to the token AND rather than to null,
    and a dangling trailing '&' leaves a dangling trailing AND.  Both are real
    in the sample ('20TH CENTURY NURSURY &').  This is the literal reading of
    the rule and is left alone deliberately -- suppressing it would mean
    inventing a stop-list the spec does not authorise.
    """
    assert normalize_name("&") == ("AND", None)
    assert normalize_name(" & ") == ("AND", None)
    assert normalize_name("20TH CENTURY NURSURY &") == ("20TH CENTURY NURSURY AND", None)


@pytest.mark.parametrize(
    "raw",
    [12345, 0, 3.14, float("nan"), float("inf"), b"ACME LLC", bytearray(b"X"),
     True, False, [], ["ACME"], {}, {"a": 1}, (1, 2), object(), set()],
)
def test_non_string_input_is_treated_as_null(raw):
    assert normalize_name(raw) == (None, None)


def test_never_raises_on_hostile_subclass():
    """A str subclass cannot steer the pipeline: every str method is unbound."""

    class Hostile(str):
        def strip(self, *a, **kw):
            raise ValueError("boom")

        def upper(self):
            raise ValueError("boom")

        def replace(self, *a, **kw):
            raise ValueError("boom")

        def split(self, *a, **kw):
            raise ValueError("boom")

        def translate(self, *a, **kw):
            raise ValueError("boom")

    assert normalize_name(Hostile("Acme Co")) == ("ACME", "CO")


def test_never_raises_on_object_with_raising_class_property():
    """isinstance() itself can throw; the guard sits inside the try for this."""

    class RaisingClass:
        @property
        def __class__(self):
            raise RuntimeError("boom")

    assert normalize_name(RaisingClass()) == (None, None)


# --------------------------------------------------------------------------
# Suffix handling
# --------------------------------------------------------------------------

def test_name_that_is_only_a_suffix():
    """No blocking key survives; name_clean is None, never ''."""
    assert normalize_name("LLC") == (None, "LLC")
    assert normalize_name("llc") == (None, "LLC")
    assert normalize_name("L.L.C.") == (None, "LLC")
    assert normalize_name("CO") == (None, "CO")
    assert normalize_name("  inc.  llc.  ") == (None, "INC LLC")


def test_suffix_stripped_only_as_a_trailing_token():
    assert normalize_name("ACME CO") == ("ACME", "CO")
    # 'CO' is a substring of COLORADO, not a token -- must survive untouched.
    assert normalize_name("COLORADO FOLIAGE INC") == ("COLORADO FOLIAGE", "INC")
    # A suffix token that is not trailing must survive inline.
    assert normalize_name("PC APARTMENTS LLC") == ("PC APARTMENTS", "LLC")
    assert normalize_name("LHM CORP ANI") == ("LHM CORP ANI", None)


def test_apostrophes_are_deleted_not_spaced():
    """FARM'S -> FARMS, not FARM S.

    Guarded here rather than only in the module doctest: a doctest sits beside
    the code it checks, so editing both together stays green.
    """
    assert normalize_name("BERARDI'S INC") == ("BERARDIS", "INC")
    assert normalize_name("O'BRIEN LLC") == ("OBRIEN", "LLC")
    assert normalize_name("GAFFORD RANCH & FARM'S INCORPORATED") == (
        "GAFFORD RANCH AND FARMS", "INCORPORATED")
    # U+2019 must behave identically to U+0027.
    assert normalize_name("O\u2019BRIEN LLC") == normalize_name("O'BRIEN LLC")


def test_multiple_trailing_suffixes_joined_in_original_order():
    assert normalize_name("TOY CO INC") == ("TOY", "CO INC")
    assert normalize_name("PIMCO LTD INC") == ("PIMCO", "LTD INC")
    assert normalize_name("M&W COMPANY INC") == ("M AND W", "COMPANY INC")


def test_dotted_and_undotted_spellings_converge():
    """The whole point of deleting periods: these must be indistinguishable.

    Each pair is anchored to a literal as well as to its partner.  Asserting only
    normalize_name(A) == normalize_name(B) would pass against a function that
    returned (None, None) unconditionally.
    """
    assert normalize_name("U.S. Bank, N.A.") == ("US BANK", "NA") == normalize_name("US BANK NA")
    assert normalize_name("HMC, L.L.C.") == ("HMC", "LLC") == normalize_name("HMC LLC")
    assert normalize_name("Damian P.C.") == ("DAMIAN", "PC") == normalize_name("DAMIAN PC")
    assert normalize_name("C&S CO.") == ("C AND S", "CO") == normalize_name("C&S Co.")


def test_suffix_token_set_is_exactly_the_brief_list():
    """Closed-set guard.

    Dropping a token was already caught by the fixture rows.  ADDING one was not,
    and that is the dangerous direction: an extra token (say DDS) is silently
    peeled out of every name that ends in it, destroying part of the blocking key
    with no test going red.
    """
    expected = {s.replace(".", "") for s in BRIEF_SUFFIX_SPELLINGS}
    assert set(normalize_module._SUFFIX_TOKENS) == expected


def test_returned_suffix_is_always_a_recognised_normalised_token():
    for raw, _clean, suffix in FIXTURE_ROWS:
        if suffix is None:
            continue
        for tok in suffix.split(" "):
            assert tok in normalize_module._SUFFIX_TOKENS, (raw, tok)


# --------------------------------------------------------------------------
# Contract invariants
# --------------------------------------------------------------------------

def test_never_returns_empty_string():
    """The invariant the fixture's empty-field-means-None convention rests on."""
    extra = ["", "   ", "...", "LLC", "L.L.C.", "&", "'", "©", "�", None]
    for raw in [r[0] for r in FIXTURE_ROWS] + extra:
        clean, suffix = normalize_name(raw)
        assert clean != "", raw
        assert suffix != "", raw


@pytest.mark.parametrize(
    "raw",
    [r[0] for r in FIXTURE_ROWS] + ["ACME CO", "LLC", "COLORADO FOLIAGE INC"],
    ids=[f"row{i:02d}" for i in range(1, len(FIXTURE_ROWS) + 1)]
        + ["acme_co", "only_suffix", "inner_co"],
)
def test_idempotency(raw):
    """Re-normalising a cleaned name is a no-op and yields no further suffix."""
    clean, _suffix = normalize_name(raw)
    assert normalize_name(clean) == (clean, None)


@pytest.mark.parametrize("raw", [r[0] for r in FIXTURE_ROWS])
def test_purity_input_is_unchanged(raw):
    """Compare against an independently constructed copy, not against itself.

    `str(raw)` returns the SAME object for a str, so asserting raw == str(raw)
    is x == x and can never fail.  Building the snapshot character-by-character
    makes it a real second object.  str immutability is what ultimately
    guarantees purity here; this asserts it rather than assuming it.
    """
    snapshot = "".join(raw)
    normalize_name(raw)
    assert raw == snapshot
    assert len(raw) == len(snapshot)


def test_no_mutable_module_state():
    """Calling the function must not perturb the module-level constants."""
    suffixes_before = set(normalize_module._SUFFIX_TOKENS)
    map_before = {k: dict(v) for k, v in COLUMN_MAP.items()}
    for raw, _c, _s in FIXTURE_ROWS:
        normalize_name(raw)
    assert set(normalize_module._SUFFIX_TOKENS) == suffixes_before
    assert COLUMN_MAP == map_before


def test_repeated_calls_are_stable():
    for raw, _c, _s in FIXTURE_ROWS:
        assert normalize_name(raw) == normalize_name(raw)


# --------------------------------------------------------------------------
# COLUMN_MAP -- the two party tables are not column-compatible
# --------------------------------------------------------------------------

SHARED_LOGICAL_KEYS = [
    "party_id", "name", "address", "city", "state", "zip",
    "file_id", "action_type", "record_status",
]


def test_column_map_has_both_tables():
    assert set(COLUMN_MAP) == {"debtors", "secured_parties"}


@pytest.mark.parametrize("table", ["debtors", "secured_parties"])
def test_column_map_has_every_shared_logical_key(table):
    for key in SHARED_LOGICAL_KEYS:
        assert key in COLUMN_MAP[table], (table, key)


def test_column_map_records_the_incompatibilities():
    assert COLUMN_MAP["debtors"]["party_id"] == "debtorid"
    assert COLUMN_MAP["secured_parties"]["party_id"] == "spid"
    # Table-specific tail columns: present on one table only.
    assert COLUMN_MAP["debtors"]["efs_unique_id"] == "efsuniqueid"
    assert "efs_unique_id" not in COLUMN_MAP["secured_parties"]
    assert COLUMN_MAP["secured_parties"]["assignor"] == "assignor"
    assert "assignor" not in COLUMN_MAP["debtors"]


def test_record_status_column_is_spelled_identically_in_both_tables():
    """Guards a claim in the brief that the real headers contradict."""
    assert (COLUMN_MAP["debtors"]["record_status"]
            == COLUMN_MAP["secured_parties"]["record_status"]
            == "recordstatus")


@pytest.mark.parametrize(
    ("table", "sample"),
    [("debtors", "sample_debtors_50k.csv"), ("secured_parties", "sample_sp_25k.csv")],
)
def test_column_map_matches_the_real_csv_headers(table, sample):
    """Every physical name must exist in the fetched sample; none invented."""
    path = _ROOT / "raw_pages" / sample
    if not path.exists():
        pytest.skip(f"{sample} not present")
    with path.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    missing = [c for c in COLUMN_MAP[table].values() if c not in header]
    assert missing == [], f"{table}: not in {sample}: {missing}"


# --------------------------------------------------------------------------
# The module docstring's examples must stay true
# --------------------------------------------------------------------------

def test_module_doctests():
    results = doctest.testmod(normalize_module, verbose=False)
    assert results.failed == 0, f"{results.failed} of {results.attempted} doctests failed"
    assert results.attempted > 0
