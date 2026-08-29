"""Pure name normalisation for the UCC-filing entity-resolution demo.

Contract
========

    normalize_name(raw: str | None) -> tuple[str | None, str | None]

Returns ``(name_clean, suffix)``.

* **Never raises.**  For *any* input whatsoever -- ``None``, ``float("nan")``,
  ``bytes``, ``int``, a list, an object with a hostile ``__str__`` -- the worst
  case is ``(None, None)``.  Anything that is not a ``str`` instance is treated
  as null.  In the fetched sample, 22,422 of 50,000 debtor rows (44.84%) have a
  blank ``organizationname``, because Colorado publishes no individual-name
  column at all: the null path is the hot path here, not the edge case.
* **Never mutates its input.**  ``str`` is immutable and nothing else is touched.
* **Pure.**  No I/O, no logging, no printing, no clock, no randomness, no
  mutable module-level state.  Same input -> same output, forever.

Why the suffix is split out
===========================

The legal-form suffix (``LLC``, ``INC``, ``CORP``, ``NA`` ...) is
**signal for comparison** and **noise for blocking**, and a single string
cannot be both:

* *Blocking* groups candidate records so they can be compared at all.  Suffixes
  are near-constant across the corpus -- keeping them inside the key drags every
  pair toward a spurious agreement on the shared tail and buries the
  discriminating head of the name.  Block on ``name_clean``.
* *Comparison* scores a candidate pair.  There the suffix is real evidence:
  ``ACME LLC`` and ``ACME INC`` are two different registered entities, and the
  suffix is the only thing that says so.  Score on ``suffix`` as its own feature.

Splitting the two at parse time is what lets the pipeline block loosely and
compare strictly, so the split lives here rather than in each consumer.

The caller keeps the untouched ``organizationname`` alongside ``name_clean``.
Nothing here destroys the original.

Normalisation rules (authoritative order of operations)
=======================================================

1. Reject null / blank / non-``str`` -> ``(None, None)``.
2. Uppercase.
3. ``&`` -> ``" AND "``.
4. ``.``, ``'`` (U+0027) and U+2019 are **deleted**, so ``L.L.C.`` becomes the
   single token ``LLC`` and ``FARM'S`` becomes ``FARMS``.
5. **Trailing registration boilerplate is removed** -- repeatedly, while commas
   and parentheses are still intact:
   * a trailing ``(THE)`` (7,475 rows, 0.250% of named parties);
   * a trailing ``, A <clause>`` **only when every token of that clause is in the
     closed legal-form/jurisdiction vocabulary** ``_FORM_VOCAB`` (44,538 rows,
     1.491%).
   The vocabulary is a *whitelist consumed to end-of-string*, never a blocklist:
   the clause must be entirely legal form.  ``OF``, ``DIVISION``, ``SUBSIDIARY``
   and every company name are absent from it, so ``, A DIVISION OF TRUIST BANK``
   and ``, A WHOLLY OWNED SUBSIDIARY OF ACCURAY INCORPORATED`` are immune **by
   construction** -- 32,013 rows whose clause names a PARENT ENTITY, which is
   real hierarchical identity and not noise.  A bare truncated ``, A`` with
   nothing after it (``HORNINGS INC., A``, ``CAPITAL ONE N,A``) does not match
   either, because a following token is required.

   Why this exists: in every one of these shapes the boilerplate sits *after* the
   legal suffix, so step 8 could never peel the suffix and the record could never
   share a blocking key with its clean twin.  ``WOODMEN ... JV LLC, A COLORADO
   LIMITED LIABILITY COMPANY`` used to block on the whole string; it now blocks
   on ``WOODMEN ... JV`` with suffix ``LLC``, exactly like ``WOODMEN ... JV LLC``.

   If a stripped clause ends in a recognised suffix token and the head yields no
   suffix of its own, that token is used as the suffix (``, A PROFESSIONAL
   CORPORATION`` -> ``CORPORATION``), so cleaning the blocking key does not throw
   away the comparison feature.

6. Every other character that is not an ASCII letter, ASCII digit or space is
   **replaced by a space** -- ``,;:-/\\()[]#"!?*+=@%`` and *all* non-ASCII
   (accented letters, U+00A9, U+FFFD, CJK ...).  Non-ASCII is 2 rows in 50,000
   (0.004%), so no confusable or transliteration map is built: a stage that
   provably never fires is padding.
7. Collapse whitespace runs to one space; trim.
8. Peel legal-form suffixes off the **right**, greedily and repeatedly, as whole
   tokens only.  ``COLORADO FOLIAGE INC`` keeps its inner ``CO``; ``ACME CO``
   does not.  Multiple trailing suffixes are all peeled and returned joined by a
   single space in original left-to-right order.  If peeling consumes every
   token, ``name_clean`` is ``None`` -- never ``""``.

Because step 4 deletes periods before step 8 matches, the dotted spellings
(``L.L.C.``, ``L.P.``, ``P.C.``, ``N.A.``) collapse onto the canonical dot-free
tokens and are recognised without a separate alias table.  The returned suffix is
therefore always the normalised dot-free form: ``L.L.C.`` yields ``"LLC"``.

Deliberately NOT done here: no phonetic key, no token-set reordering, and no
*blocklist* of noise words -- step 5 is a closed whitelist, which fails safe
(an unrecognised clause is kept) where a blocklist fails destructive.  DBA /
D/B/A splitting (0.94% of rows) is also NOT done: extracting two entities from
one string is a different feature, not a normalisation tweak.  Blocking-recall loss is measured downstream,
not papered over in the normaliser.
"""

import re
from typing import Final

__all__ = ["COLUMN_MAP", "normalize_name"]


# --------------------------------------------------------------------------
# Character handling
# --------------------------------------------------------------------------

#: Deleted outright (replaced by nothing) so that dotted suffixes collapse into
#: one token and possessives close up: "L.L.C." -> "LLC", "FARM'S" -> "FARMS".
_DELETED_CHARS: Final[dict[int, None]] = {
    0x002E: None,  # FULL STOP
    0x0027: None,  # APOSTROPHE
    0x2019: None,  # RIGHT SINGLE QUOTATION MARK
}

#: Everything that is not an ASCII letter, ASCII digit or space becomes a space.
#: The class is deliberately ASCII-only: non-ASCII letters are dropped, not
#: transliterated.  It also catches \t, \n, \r and NUL, which step 6 collapses.
#:
#: a-z is included even though step 2 has already uppercased the string.  No
#: codepoint currently uppercases to a lowercase ASCII letter, so the lowercase
#: half is unreachable today -- but writing the class as literally stated in the
#: rules keeps it independent of step 2.  Were the two steps ever reordered, the
#: failure mode is a soft pass-through rather than "acme llc" -> (None, None).
_TO_SPACE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9 ]")

#: Legal-form suffixes, in the normalised (dot-free, uppercase) form the cleaning
#: pipeline produces.  Every raw spelling named in the brief lands on one of
#: these 14 tokens:
#:     LLC, L.L.C.       -> LLC           LP, L.P. -> LP
#:     INC               -> INC           LLP      -> LLP
#:     INCORPORATED      -> INCORPORATED  PLLC     -> PLLC
#:     CORP              -> CORP          PC, P.C. -> PC
#:     CORPORATION       -> CORPORATION   NA, N.A. -> NA
#:     CO                -> CO
#:     COMPANY           -> COMPANY
#:     LTD               -> LTD
#:     LIMITED           -> LIMITED
#: INC/INCORPORATED, CORP/CORPORATION, CO/COMPANY and LTD/LIMITED are kept
#: distinct rather than folded together: the suffix is a comparison feature, and
#: folding them would discard the spelling difference before the model sees it.
_SUFFIX_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "LLC",
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "LTD",
        "LIMITED",
        "LP",
        "LLP",
        "PLLC",
        "PC",
        "NA",
    }
)


#: Closed legal-form and jurisdiction vocabulary for step 5.  A trailing
#: ``, A <clause>`` is removed ONLY when EVERY token of the clause appears here.
#: Deliberately absent: ``OF``, ``DIVISION``, ``SUBSIDIARY``, ``WHOLLY``,
#: ``OWNED``, ``PROGRAM``, ``FORMERLY``, ``KNOWN``, and every company name.
#: Their absence is what makes ``, A DIVISION OF NBH BANK`` immune -- the rule
#: cannot be tricked into eating a parent entity, because it only ever eats words
#: it already knows to be legal form.  Derived from the full 2,987,031-row named
#: party set, not from a sample: it covers 58.2% of observed trailing clauses and
#: leaves the other 41.8% -- which name parents -- untouched.
_FORM_VOCAB: Final[frozenset[str]] = frozenset("""
CORPORATION CORP COMPANY CO INCORPORATED INC LIMITED LIABILITY PARTNERSHIP LP LLP LLC
GENERAL PROFESSIONAL BANKING BANK SAVINGS LOAN ASSOCIATION NATIONAL FEDERAL FEDERALLY
CHARTERED STATE STOCK CREDIT UNION AND NON PROFIT NONPROFIT ORGANIZATION CORPORATE BODY
COOPERATIVE MUTUAL TRUST PUBLIC PRIVATE FOREIGN DOMESTIC BENEFIT SOLE PROPRIETORSHIP
JOINT VENTURE UNINCORPORATED
COLORADO DELAWARE CALIFORNIA TEXAS WYOMING NEVADA NEBRASKA FLORIDA KANSAS UTAH ARIZONA
MONTANA MINNESOTA MISSOURI ILLINOIS IOWA OKLAHOMA OHIO GEORGIA VIRGINIA MICHIGAN WISCONSIN
INDIANA NEW YORK JERSEY MEXICO HAMPSHIRE DAKOTA SOUTH NORTH WEST EAST CAROLINA PENNSYLVANIA
MARYLAND MASSACHUSETTS OREGON WASHINGTON IDAHO ARKANSAS TENNESSEE KENTUCKY ALABAMA LOUISIANA
CONNECTICUT RHODE ISLAND MAINE VERMONT ALASKA HAWAII MISSISSIPPI US USA UNITED STATES
""".split())

#: A trailing "(THE)" -- a definite article wrapped onto the end of the name.
_PAREN_THE: Final[re.Pattern[str]] = re.compile(r"\(\s*THE\s*\)\s*$")

#: A trailing ", A <clause>".  ``\S`` after ``\s+`` REQUIRES a following token, so a
#: bare truncated ", A" never matches.  ``.*$`` is greedy to end-of-string, so the
#: ENTIRE tail must clear the vocabulary -- a clause with a trailing
#: ", OR ITS SUCCESSORS, ASSIGNS" fails the check and is kept whole.
_TRAILING_A: Final[re.Pattern[str]] = re.compile(r",\s*A\s+(\S.*)$")

#: Word extraction for the vocabulary test, so punctuation inside a clause
#: ("NON-PROFIT") does not defeat it.
_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9]+")


def _strip_boilerplate(text: str) -> tuple[str, str | None]:
    """Apply step 5.  Returns ``(text, fallback_suffix)``.

    ``fallback_suffix`` is the removed clause's trailing legal-form token when it
    is one, so that a name whose ONLY suffix lived inside the boilerplate keeps
    that comparison feature after the blocking key is cleaned.
    """
    fallback: str | None = None
    while True:
        m = _PAREN_THE.search(text)
        if m:
            text = str.rstrip(text[: m.start()])
            continue
        m = _TRAILING_A.search(text)
        if m:
            toks = _WORD.findall(m.group(1))
            if toks and all(t in _FORM_VOCAB for t in toks):
                if fallback is None and toks[-1] in _SUFFIX_TOKENS:
                    fallback = toks[-1]
                text = str.rstrip(text[: m.start()])
                continue
        return text, fallback


def _tokenize(text: str) -> tuple[list[str], str | None]:
    """Apply steps 3-6 to an already-uppercased string; return its tokens.

    Every ``str`` method is called unbound so that a ``str`` subclass with
    overridden methods (``numpy.str_``, a user type) cannot steer the pipeline.
    """
    text = str.replace(text, "&", " AND ")
    text = str.translate(text, _DELETED_CHARS)
    # Step 5 runs HERE -- after periods are gone (so "INC., A" reads as "INC, A")
    # but before punctuation becomes space (so the comma and parens it anchors on
    # still exist).
    text, fallback = _strip_boilerplate(text)
    text = _TO_SPACE.sub(" ", text)
    # Only U+0020 survives the substitution, so split() both collapses runs of
    # whitespace and trims the ends -- step 7 in one call.
    return str.split(text), fallback


def _peel_suffixes(tokens: list[str]) -> tuple[str | None, str | None]:
    """Apply step 7.  ``tokens`` is non-empty on entry and is never mutated."""
    cut = len(tokens)
    while cut > 0 and tokens[cut - 1] in _SUFFIX_TOKENS:
        cut -= 1
    head = tokens[:cut]
    tail = tokens[cut:]
    return (" ".join(head) if head else None, " ".join(tail) if tail else None)


def normalize_name(raw: str | None) -> tuple[str | None, str | None]:
    """Normalise an organisation name into ``(name_clean, suffix)``.

    Block on ``name_clean``; score on ``suffix``.  Either element may be
    ``None``; neither is ever the empty string.  This function never raises and
    never mutates its argument.

    >>> normalize_name("Acme Co")
    ('ACME', 'CO')
    >>> normalize_name("Colorado Foliage, Inc.")
    ('COLORADO FOLIAGE', 'INC')
    >>> normalize_name("Smith & Sons Farm's L.L.C.")
    ('SMITH AND SONS FARMS', 'LLC')
    >>> normalize_name("Jones D.M.D.,P.C.")
    ('JONES DMD', 'PC')
    >>> normalize_name("Acme Holdings CO., LLC")
    ('ACME HOLDINGS', 'CO LLC')
    >>> normalize_name("Bank of the West N.A. Co.")
    ('BANK OF THE WEST', 'NA CO')
    >>> normalize_name("austin-nollsch")
    ('AUSTIN NOLLSCH', None)
    >>> normalize_name("160 Corp")
    ('160', 'CORP')

    Trailing registration boilerplate is removed so the real suffix can be seen:

    >>> normalize_name("YM LLC, A COLORADO LIMITED LIABILITY COMPANY")
    ('YM', 'LLC')
    >>> normalize_name("Door Co (THE)")
    ('DOOR', 'CO')
    >>> normalize_name("Dennis L. Burgner, D.D.S.,A Professional Corporation")
    ('DENNIS L BURGNER DDS', 'CORPORATION')

    ...but a clause naming a PARENT ENTITY is hierarchical identity, not noise,
    and survives untouched -- the vocabulary contains no "OF" and no "DIVISION":

    >>> normalize_name("SPRS, A DIVISION OF CSC")
    ('SPRS A DIVISION OF CSC', None)
    >>> normalize_name("HORNINGS INC., A")
    ('HORNINGS INC A', None)

    Non-ASCII letters are dropped rather than transliterated, which can split a
    token; at 2 rows in 50,000 that is accepted rather than mapped:

    >>> normalize_name("Caf\\xe9 Fran\\xe7ais Ltd")
    ('CAF FRAN AIS', 'LTD')

    A name that is nothing but legal form yields no blocking key at all:

    >>> normalize_name("L.L.C.")
    (None, 'LLC')
    >>> normalize_name("  inc.  llc.  ")
    (None, 'INC LLC')

    Blank, punctuation-only, and non-string inputs are all null:

    >>> normalize_name("   ")
    (None, None)
    >>> normalize_name("...")
    (None, None)
    >>> normalize_name(None)
    (None, None)
    >>> normalize_name(float("nan"))
    (None, None)
    >>> normalize_name(b"ACME LLC")
    (None, None)
    """
    try:
        # Step 1.  Covers None, NaN, ints, bytes, lists, pandas.NA, ...
        # bool is not a str either, so True -> (None, None).
        #
        # This guard is deliberately INSIDE the try.  Hoisting it above looks
        # tidier but an object whose `__class__` is a raising property then
        # escapes as RuntimeError, breaking the never-raises contract.
        if not isinstance(raw, str):
            return (None, None)

        # Fast path for the ~45% of rows that are blank: skip the translate,
        # the regex sub and the split entirely.  The post-tokenise check below
        # is still required -- it catches punctuation-only and non-ASCII-only
        # rows, which survive strip() but tokenise to nothing.
        if not str.strip(raw):
            return (None, None)

        # Step 2, called unbound for the same subclass reason as _tokenize.
        tokens, fallback = _tokenize(str.upper(raw))
        if not tokens:  # punctuation-only or non-ASCII-only
            return (None, None)

        head, tail = _peel_suffixes(tokens)
        if tail is None and fallback is not None:
            tail = fallback
        return (head, tail)
    except Exception:
        # Last line of defence.  This function sits on the hot path of every row
        # in the pipeline and the contract is "never raises", so an unforeseen
        # input degrades to a null rather than killing the run.  BaseException
        # (KeyboardInterrupt, SystemExit) is deliberately NOT caught.
        return (None, None)


# --------------------------------------------------------------------------
# Physical schema
# --------------------------------------------------------------------------

#: The two party tables are NOT column-compatible: the party-id column differs
#: (``debtorid`` vs ``spid``) and the tail column differs (``efsuniqueid`` vs
#: ``assignor``).  Every other physical header is identical -- INCLUDING
#: ``recordstatus``, which is spelled the same in both tables.
#:
#: Logical key -> physical column, per table.  All 20 names verified against the
#: headers of raw_pages/sample_debtors_50k.csv and raw_pages/sample_sp_25k.csv;
#: nothing here is invented.
#:
#: ``efs_unique_id`` exists only under ``debtors`` and ``assignor`` only under
#: ``secured_parties``, so any code that walks both tables generically must use
#: ``.get()`` rather than ``[]`` for those two keys.  The shared keys --
#: ``party_id`` through ``record_status`` -- are present for both.
#:
#: ``normalize_name`` is shared by both tables; this row-level map is what is
#: not.  Read-only: treat as a constant, never mutate in place.
COLUMN_MAP: Final[dict[str, dict[str, str]]] = {
    "debtors": {
        "party_id": "debtorid",
        "name": "organizationname",
        "address": "address1",
        "city": "city",
        "state": "state",
        "zip": "zipcode",
        "file_id": "fileid",
        "action_type": "actiontype",
        "record_status": "recordstatus",
        "efs_unique_id": "efsuniqueid",
    },
    "secured_parties": {
        "party_id": "spid",
        "name": "organizationname",
        "address": "address1",
        "city": "city",
        "state": "state",
        "zip": "zipcode",
        "file_id": "fileid",
        "action_type": "actiontype",
        "record_status": "recordstatus",
        "assignor": "assignor",
    },
}
