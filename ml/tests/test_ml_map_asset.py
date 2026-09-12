"""The map asset obeys the same rules as the Lab script (plan C Task 5's asset, map added alongside).

ml-lab.js is scanned for HTML-parsing sinks by its own tests. docs/assets/ml-map.js is a second
published script on the same page and nothing scanned it, so the guarantee was a sentence in a
comment rather than a check. It is a check now.

The map lives in its own file and its own container deliberately: ml-lab.js publishes an exact list
of six section headings that its node tests assert on, so a seventh section rendered there would
fail them.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAP_SCRIPT = REPO / "docs" / "assets" / "ml-map.js"
MAP_STYLE = REPO / "docs" / "assets" / "ml-map.css"
PAGE = REPO / "docs" / "ml" / "index.html"
MAP_DATA = REPO / "docs" / "data" / "ml" / "map_zips.json"


def test_the_map_script_never_parses_html():
    source = MAP_SCRIPT.read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in source, sink


def test_the_page_loads_the_map_with_project_relative_paths():
    html = PAGE.read_text(encoding="utf-8")
    assert '<link rel="stylesheet" href="../assets/ml-map.css">' in html
    assert '<script src="../assets/ml-map.js"></script>' in html
    assert "window.mountMLMap(document.getElementById('ml-map')" in html
    assert re.search(r'(?:src|href)="/', html) is None, "root-relative paths break under the /UCC-Demo/ prefix"


def test_the_map_keeps_the_lab_section_count_untouched():
    """The map must not render into the Lab's container: ml-lab.js's node tests assert exactly six
    h2 headings there, so a seventh would fail them."""
    html = PAGE.read_text(encoding="utf-8")
    assert '<section id="ml-map"' in html
    assert html.index('<main id="ml-lab">') < html.index('<section id="ml-map"')
    # The intent is that the script never REACHES FOR the Lab's container -- it is handed its own.
    # Asserting the substring "ml-lab" is absent was wrong: the file's comment explains at length why
    # the map is separate from ml-lab.js, so the test failed on its own documentation.
    source = MAP_SCRIPT.read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = "\n".join(re.sub(r"//.*$", "", line) for line in code.splitlines())
    assert "getElementById" not in code, "the map is handed its container; it must not look one up"
    assert "ml-lab" not in code, "the map script must not name the Lab's container in code"


def test_the_map_reuses_the_demo_projection_and_palette():
    """A postcode has to land in the same place on both maps, so the bbox, ring and padding are the
    map page's own, not re-derived."""
    source = MAP_SCRIPT.read_text(encoding="utf-8")
    demo = (REPO / "docs" / "index.html").read_text(encoding="utf-8")
    for constant in ("-109.0448", "-102.0415", "36.9930", "41.0034", "-73.7278", "42.0500"):
        assert constant in source and constant in demo, constant
    assert "var PAD = 14" in source and "const PAD = 14" in demo


def test_the_map_data_is_generated_and_complete():
    import json

    doc = json.loads(MAP_DATA.read_text(encoding="utf-8"))
    assert doc["release_id"], "the map must name the release it describes"
    totals = doc["totals"]
    assert totals["both"] + totals["model_only"] == totals["model_suggest"]
    assert totals["both"] + totals["rules_only"] == totals["rules_accept"]
    assert totals["zips"] == len(doc["zips"]) > 1000
    assert sum(z["n"] for z in doc["zips"]) == totals["filings"]
    for z in doc["zips"]:
        assert z["b"] + z["om"] == z["s"] and z["b"] + z["orl"] == z["a"], z
        assert -180 <= z["lon"] <= 0 and 0 <= z["lat"] <= 90, z
