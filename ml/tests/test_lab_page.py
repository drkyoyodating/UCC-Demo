"""The static ML Lab page (plan C Task 5): project-relative paths, no HTML-parsing sinks, the map's
palette, the node unit tests, the SYNTHETIC preview tool, and the contract between the exported
public JSON and the paths docs/assets/ml-lab.js reads. Synthetic data only; the last test checks the
committed public data once Task 10 has published it and skips before then."""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ucc_ml.contracts import LABEL_DISCLOSURE
from ucc_ml.labeling import pooled_disclosure

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "ml" / "index.html"
SCRIPT = REPO / "docs" / "assets" / "ml-lab.js"
STYLE = REPO / "docs" / "assets" / "ml-lab.css"
MAP_PAGE = REPO / "docs" / "index.html"
NODE_TEST = REPO / "ml" / "tests" / "js" / "test_ml_lab.cjs"
PALETTE = ("--co", "--ct", "--co-dim", "--ct-dim", "--ink", "--mut", "--line", "--bg")


def expected_provenance(metrics: dict) -> str:
    """What every public document of this release must disclose, rebuilt from the release's own
    round map with the same helper that writes it.

    A literal here would be a FALSE expectation for the real labels, whose two rounds were labelled
    under different arrangements: main_v1 retains its disagreements as unresolved, pilot_v1 was
    founder-adjudicated, and the published sentence names both."""
    labels = metrics["test"]["labels"]
    by_round = labels.get("disclosure_by_round")
    return pooled_disclosure(by_round) if by_round else LABEL_DISCLOSURE


def js_paths(name: str) -> list[str]:
    """The `var <NAME>_PATHS = [...]` list in ml-lab.js: the single source of what the page reads."""
    match = re.search(rf"var {name}_PATHS = \[(.*?)\];", SCRIPT.read_text(encoding="utf-8"), re.S)
    assert match, f"{name}_PATHS not found in {SCRIPT}"
    return re.findall(r"'([^']+)'", match.group(1))


def missing_paths(doc, paths: list[str]) -> list[str]:
    """Python twin of missingPaths() in ml-lab.js: `a.b[].c` needs a non-empty list at a.b whose items all have c."""

    def has(value, parts: list[str]) -> bool:
        if not parts:
            return True
        head, rest = parts[0], parts[1:]
        is_list = head.endswith("[]")
        key = head[:-2] if is_list else head
        if not isinstance(value, dict) or key not in value:
            return False
        child = value[key]
        if not is_list:
            return has(child, rest) if rest else True
        if not isinstance(child, list) or not child:
            return False
        return all(has(item, rest) for item in child)

    return [path for path in paths if not has(doc, path.split("."))]


def css_variables(block: str) -> dict[str, str]:
    return {name: value.strip() for name, value in re.findall(r"(--[a-z-]+):\s*([^;]+);", block)}


def load_preview_tool():
    spec = importlib.util.spec_from_file_location("make_lab_preview", REPO / "ml" / "tools" / "make_lab_preview.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_page_loads_project_relative_assets_and_data():
    html = PAGE.read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in html
    assert '<link rel="stylesheet" href="../assets/ml-lab.css">' in html
    assert '<script src="../assets/ml-lab.js"></script>' in html
    assert "window.mountMLLab(document.getElementById('ml-lab'), { dataBase: '../data/ml/' });" in html
    assert re.search(r'(?:src|href)="/', html) is None, "root-relative paths break under the /UCC-Demo/ prefix"
    assert "apiUrl" not in html, "the API URL comes from the manifest, never from the page"


def test_script_never_parses_html_and_discloses_label_provenance():
    source = SCRIPT.read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in source, sink
    # The page must disclose what the RELEASE states, not a sentence the script carries: it reads
    # label_provenance and renders disclosure_by_round whenever the rounds differ. Asserting the
    # literal appears in the source would pin exactly the bug this replaces.
    assert "label_provenance" in source
    assert "disclosure_by_round" in source


def test_style_reuses_the_map_palette_exactly():
    root_block = re.search(r":root\{(.*?)\}", MAP_PAGE.read_text(encoding="utf-8"), re.S).group(1)
    lab_block = re.search(r"^\.mll\{(.*?)\}", STYLE.read_text(encoding="utf-8"), re.S | re.M).group(1)
    map_vars, lab_vars = css_variables(root_block), css_variables(lab_block)
    for name in PALETTE:
        assert lab_vars.get(name) == map_vars[name], name


def test_the_four_path_lists_are_present():
    assert "releases[].api_url" in js_paths("MANIFEST")
    assert "test.labels.repeat_consistency" in js_paths("METRICS")
    assert "examples[].precomputed" in js_paths("EXAMPLES")
    assert "sections[].text" in js_paths("MODEL_CARD")


def test_missing_paths_matches_the_script_semantics():
    doc = {"a": [{"b": 1}, {"b": None}], "c": {"d": 0}, "e": []}
    assert missing_paths(doc, ["a[].b", "c.d", "c", "e[].x", "a[].z", "f"]) == ["e[].x", "a[].z", "f"]


def test_node_unit_tests_pass():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed here (CI installs node 22)")
    check = subprocess.run([node, "--check", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stderr
    run = subprocess.run([node, "--test", str(NODE_TEST)], capture_output=True, text=True, cwd=REPO, check=False)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]


def test_preview_refuses_to_write_under_docs():
    tool = load_preview_tool()
    with pytest.raises(ValueError, match="outside docs/"):
        tool.make_preview(REPO / "docs" / "ml-preview", repo_root=REPO, bundle_dir=Path("never-used"))
    assert not (REPO / "docs" / "ml-preview").exists()


@pytest.fixture(scope="module")
def preview_site(synthetic_release_dir, tmp_path_factory):
    return load_preview_tool().make_preview(tmp_path_factory.mktemp("lab-preview"), repo_root=REPO,
                                            bundle_dir=synthetic_release_dir)


def test_preview_copies_the_page_and_marks_the_manifest_synthetic(preview_site, synthetic_release_dir):
    assert (preview_site / "ml" / "index.html").read_bytes() == PAGE.read_bytes()
    assert (preview_site / "assets" / "ml-lab.js").read_bytes() == SCRIPT.read_bytes()
    assert (preview_site / "assets" / "ml-lab.css").read_bytes() == STYLE.read_bytes()
    manifest = json.loads((preview_site / "data" / "ml" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["preview"] is True
    assert manifest["current"] == synthetic_release_dir.name
    assert not preview_site.resolve().is_relative_to((REPO / "docs").resolve())


def test_exported_documents_carry_every_path_the_page_reads(preview_site):
    base = preview_site / "data" / "ml"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert missing_paths(manifest, js_paths("MANIFEST")) == []
    release = next(r for r in manifest["releases"] if r["release_id"] == manifest["current"])
    documents = {
        "METRICS": json.loads((base / release["metrics_url"]).read_text(encoding="utf-8")),
        "EXAMPLES": json.loads((base / release["examples_url"]).read_text(encoding="utf-8")),
        "MODEL_CARD": json.loads((base / release["model_card_url"]).read_text(encoding="utf-8")),
    }
    for name, document in documents.items():
        assert missing_paths(document, js_paths(name)) == [], name
        assert document["release_id"] == manifest["current"], name
        assert document["label_provenance"] == expected_provenance(documents["METRICS"]), name
    assert documents["EXAMPLES"]["curated"] is True and documents["EXAMPLES"]["precomputed"] is True


def test_committed_public_data_is_real_and_complete():
    """Checks docs/data/ml once plan C Task 10 has published a release; skips before then."""
    base = REPO / "docs" / "data" / "ml"
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("docs/data/ml/manifest.json is published by plan C Task 10")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "preview" not in manifest, "a SYNTHETIC preview manifest must never be committed under docs/"
    assert missing_paths(manifest, js_paths("MANIFEST")) == []
    for release in manifest["releases"]:
        texts = {key: (base / release[key]).read_text(encoding="utf-8") for key in ("metrics_url", "examples_url", "model_card_url")}
        for key, text in texts.items():
            # Path markers without a trailing slash: they must survive Task 1 Step 8's sanitiser.
            for private in ("/Users", "/home", "/private", "frozen_dir"):
                assert private not in text, (key, private)
        metrics, examples, card = (json.loads(texts[key]) for key in ("metrics_url", "examples_url", "model_card_url"))
        assert missing_paths(metrics, js_paths("METRICS")) == []
        assert missing_paths(examples, js_paths("EXAMPLES")) == []
        assert missing_paths(card, js_paths("MODEL_CARD")) == []
        assert (metrics["label_provenance"] == examples["label_provenance"] == card["label_provenance"]
                == expected_provenance(metrics))
        assert metrics["test"]["labels"]["disclosure"] == expected_provenance(metrics)
        assert metrics["release_id"] == examples["release_id"] == card["release_id"] == release["release_id"]
        assert 50 <= len(examples["examples"]) <= 100
        assert all(e["curated"] is True and e["precomputed"] is True for e in examples["examples"])


def test_map_page_links_to_the_lab_once_in_its_footer_and_outside_the_tabs():
    html = MAP_PAGE.read_text(encoding="utf-8")
    assert html.count('href="./ml/"') == 1
    link = html.index('<a href="./ml/">')
    foot_start = html.index('<p id="foot">')
    assert foot_start < link < html.index("</p>", foot_start), "the link sits in the footer paragraph"
    tabs_start = html.index('<div id="tabs">')
    assert not tabs_start < link < html.index("</div>", tabs_start), "the link is not a tab"
    assert 'data-p="ml"' not in html
    assert html.count('class="chip') == 6, "no new chip"
