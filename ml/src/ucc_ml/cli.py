"""Command line for the UCC ML Lab.

    ./.venv-ml/bin/python -m ucc_ml.cli <command> --config ml/configs/v1.yaml

Every subcommand is a thin wrapper over a tested function in another module; nothing is computed
here. Two marker comments are the anchors every plan edits against, and neither is ever changed
(contract K11):

* inside ``build_parser`` a plan replaces the subcommand marker line with
  ``sp = sub.add_parser("<name>", help="...")``, ``_add_config_arg(sp)``,
  ``sp.set_defaults(func=cmd_<name>)`` followed by the same marker line, unchanged;
* above ``main`` a plan replaces the cmd marker line with ``def cmd_<name>(ns) -> int:`` followed by
  the same marker line, unchanged. A ``cmd_<name>`` imports its implementation inside the function,
  so ``--help`` never imports pandas, pydantic, sklearn, mlflow or fastapi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ucc_ml import __version__

#: The labelling rounds every --round argument accepts. Spelled here rather than imported from
#: ucc_ml.contracts because building the parser must not import pydantic (contract K11, pinned by
#: test_building_the_parser_imports_nothing_heavy). test_round_choices_match_the_contract asserts
#: this tuple equals contracts.LABELLING_ROUNDS, so the duplication cannot drift.
ROUND_CHOICES: tuple[str, ...] = ("pilot_v1", "ablation_v1", "yield_probe_v1", "main_v1", "queue_v1")


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True, type=Path,
        help="path to the run config, e.g. ml/configs/v1.yaml",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ucc_ml",
        description="UCC ML Lab: candidate dataset, blind labels, splits, model, evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"ucc_ml {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sp = sub.add_parser("build-candidates", help="snapshot Parquet -> candidates.parquet + reconciliation")
    _add_config_arg(sp)
    sp.set_defaults(func=cmd_build_candidates)
    sp = sub.add_parser("make-pilot", help="draw the 200-case development pilot (per_stratum x 4 strata)")
    _add_config_arg(sp)
    sp.set_defaults(func=cmd_make_pilot)
    sp = sub.add_parser("freeze-splits", help="grouped 65/15/20 splits by group_id; every pilot group in train")
    _add_config_arg(sp)
    sp.set_defaults(func=cmd_freeze_splits)
    sp = sub.add_parser("label-blind", help="emit a round's chunked blind queue, its private key and the pre-registration digest")
    _add_config_arg(sp)
    sp.add_argument("--round", required=True, choices=ROUND_CHOICES)
    sp.add_argument("--chunk-size", type=int, default=None, help="cases per chunk before repeats (default: labelling.chunk_size)")
    sp.set_defaults(func=cmd_label_blind)
    sp = sub.add_parser("labeller-brief", help="render one queue chunk's blind labeller brief (policy and chunk inline, K14)")
    _add_config_arg(sp)
    sp.add_argument("--round", required=True, choices=ROUND_CHOICES)
    sp.add_argument("--part", required=True, type=int)
    sp.set_defaults(func=cmd_labeller_brief)
    sp = sub.add_parser("labelling-status", help="which queue chunks each blind pass has returned")
    _add_config_arg(sp)
    sp.add_argument("--round", required=True, choices=ROUND_CHOICES)
    sp.set_defaults(func=cmd_labelling_status)
    sp = sub.add_parser("write-raw-labels", help="validate one labeller agent's structured output and write its raw CSV")
    _add_config_arg(sp)
    sp.add_argument("--round", required=True, choices=ROUND_CHOICES)
    sp.add_argument("--pass", dest="pass_letter", required=True, choices=("a", "b"))
    sp.add_argument("--part", required=True, type=int)
    sp.add_argument("--structured", required=True, type=Path, help="the agent's structured output, saved verbatim as JSON")
    sp.set_defaults(func=cmd_write_raw_labels)
    sp = sub.add_parser("import-labels", help="validate and de-alias both blind passes of a round; freeze their digests")
    _add_config_arg(sp)
    sp.add_argument("--round", required=True, choices=ROUND_CHOICES)
    sp.set_defaults(func=cmd_import_labels)
    sp = sub.add_parser("review-workbook", help="pass agreement, repeat consistency and the founder review workbook for a round")
    _add_config_arg(sp)
    sp.add_argument("--round", required=True, choices=ROUND_CHOICES)
    sp.set_defaults(func=cmd_review_workbook)
    sp = sub.add_parser("import-founder-review", help="import the founder's review workbook of a round; freeze its digest")
    _add_config_arg(sp)
    sp.add_argument("--round", required=True, choices=ROUND_CHOICES)
    sp.set_defaults(func=cmd_import_founder_review)
    sp = sub.add_parser("validate-labels", help="merge every validated round into labels.csv + labels_manifest.json (refuses undecided disagreements)")
    _add_config_arg(sp)
    sp.set_defaults(func=cmd_validate_labels)
    sp = sub.add_parser("build-screen", help="fit the sampling screen and assign every case a cell")
    _add_config_arg(sp)
    sp.set_defaults(func=cmd_build_screen)
    sp = sub.add_parser("make-main-round", help="draw the screened main round per (split, stratum, cell)")
    _add_config_arg(sp)
    sp.add_argument("--dry-run", action="store_true", help="print the allocation and draw nothing")
    sp.set_defaults(func=cmd_make_main_round)
    # --- subcommands are registered below this line by later tasks (keep alphabetical) ---
    return parser


def cmd_build_candidates(ns: argparse.Namespace) -> int:
    from ucc_ml import dataset, legacy
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.provenance import git_head

    cfg = load_config(ns.config)
    paths = artefact_paths(cfg)
    vendor = legacy.verify_vendor_hashes()
    manifest = dataset.build_candidates(
        snapshot_dir=cfg.path("snapshot_dir"), out_dir=cfg.path("candidates_dir"),
        dataset_version=cfg.version.dataset_version, year_min=cfg.eligibility.year_min,
        junk=cfg.eligibility.junk_addresses,
        provenance={"config_sha256": cfg.config_sha256, "config_path": str(cfg.config_path),
                    "git_head": git_head(cfg.repo_root), "vendor": vendor},
    )
    c = manifest["candidates"]
    print(f"cases={c['rows']} by_region={c['by_region']} strata={c['by_stratum']} routes={c['by_route']}")
    print(f"parity: {'OK' if manifest['parity_ok'] else 'FAILED -- read reconciliation.json'}")
    print(f"wrote {paths.candidates_parquet} sha256={c['sha256']}")
    return 0 if manifest["parity_ok"] else 1


def cmd_make_pilot(ns: argparse.Namespace) -> int:
    import hashlib

    from ucc_ml import dataset, sampling
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.provenance import git_head, sha256_file, utc_now_iso, write_json

    cfg = load_config(ns.config)
    paths = artefact_paths(cfg)
    cases = dataset.read_candidates(paths.candidates_parquet)
    pilot = sampling.make_pilot(cases, per_stratum=cfg.sampling.pilot_per_stratum, seed=cfg.seed, purpose="pilot_v1")
    sha = dataset.write_frame(pilot, paths.pilot_cases)
    strata = {s: {"N_h": int(g.N_h.iloc[0]), "pool_h": int(g.pool_h.iloc[0]), "n_h": int(g.n_h.iloc[0]),
                  "inclusion_probability": float(g.inclusion_probability.iloc[0])}
              for s, g in pilot.groupby("sampling_stratum", sort=False)}
    manifest = {
        "purpose": "pilot_v1", "dataset_version": cfg.version.dataset_version, "seed": cfg.seed,
        "per_stratum": cfg.sampling.pilot_per_stratum, "strata": strata, "rows": int(len(pilot)),
        "by_region": {k: int(v) for k, v in sorted(pilot.region.value_counts().items())},
        "case_id_digest": hashlib.sha256(("\n".join(sorted(pilot.case_id)) + "\n").encode("utf-8")).hexdigest(),
        "pilot_cases_sha256": sha, "candidates_sha256": sha256_file(paths.candidates_parquet),
        "created_at": utc_now_iso(), "git_head": git_head(cfg.repo_root), "config_sha256": cfg.config_sha256,
    }
    write_json(paths.pilot_manifest, manifest)
    print(f"pilot rows={len(pilot)} by_region={manifest['by_region']} strata={ {k: v['n_h'] for k, v in strata.items()} }")
    print(f"wrote {paths.pilot_cases} sha256={sha}")
    return 0


def _splits_digest_manifest(parquet_sha: str, split_digest: str) -> str:
    """The published splits_v1.sha256: one checkable entry, then '# ' comments.

    A published manifest must not name a path that does not exist. This file is consumed with
    `shasum -a 256 -c`, so every non-comment line is a claim that the named file sits beside it. The
    split digest is a CONTENT digest over the assignment -- sha256 of the sorted
    'case_id,group_id,split' lines -- not the sha256 of any file, and no splits.digest file has ever
    existed in this repository. Emitting it in the checkable-entry position (with the explanation
    stuffed into the filename column) made `shasum -a 256 -c` fail on the published artefact: it
    reported splits.digest as missing, which reads to a reader as a failed integrity check of the
    splits themselves. It is therefore a comment carrying the command that reproduces it.

    Entries-then-comments matches labeling.write_digest_file, which writes every other published
    .sha256 in docs/data/ml. Kept here as a function, not an inline f-string, so the published file
    can be rewritten from the frozen artefacts without re-running the draw (the splits are frozen:
    cmd_freeze_splits redraws them and rewrites splits.parquet).
    """
    return (
        f"{parquet_sha}  splits.parquet\n"
        "# splits.parquet is the only verifiable entry in this file.\n"
        f"# split digest: {split_digest}\n"
        "#   sha256 over the sorted 'case_id,group_id,split' lines -- a content digest of the split\n"
        '#   ASSIGNMENT, not of a file; it is also split_manifest.json "digest". It is a comment\n'
        "#   because this manifest is checked with 'shasum -a 256 -c', which must not be handed a\n"
        "#   path that does not exist. Reproduce it with:\n"
        "#   python -c \"from ucc_ml.splitting import read_splits, split_digest; "
        "print(split_digest(read_splits('ml/data/splits/v1/splits.parquet')))\"\n"
        "# Committed when the splits were frozen, before any label exists. Groups are region-scoped borrower\n"
        "# names; every group containing a pilot case is in train (contract K12).\n"
    )


def cmd_freeze_splits(ns: argparse.Namespace) -> int:
    from ucc_ml import dataset, splitting
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.provenance import git_head, sha256_file, write_json

    cfg = load_config(ns.config)
    paths = artefact_paths(cfg)
    cases = dataset.read_candidates(paths.candidates_parquet)
    pilot = dataset.read_frame(paths.pilot_cases)
    ratios = {"train": cfg.splits.train, "validation": cfg.splits.validation, "test": cfg.splits.test}
    splits, manifest = splitting.freeze_splits(cases, pilot.case_id, ratios, cfg.seed,
                                               cfg.version.label_policy_version, sha256_file(paths.candidates_parquet))
    parquet_sha = dataset.write_frame(splits, paths.splits_parquet)
    manifest.update({"splits_parquet_sha256": parquet_sha, "git_head": git_head(cfg.repo_root),
                     "config_sha256": cfg.config_sha256})
    write_json(paths.split_manifest, manifest)
    digest_path = paths.public_data_dir / "splits_v1.sha256"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(_splits_digest_manifest(parquet_sha, manifest["digest"]), encoding="utf-8")
    g, c, a = manifest["groups"], manifest["cases"], manifest["audit"]
    print(f"groups={g['total']} by_split={g['by_split']} pilot_groups={g['pilot']}")
    print(f"cases={c['total']} by_split={c['by_split']} pilot_by_split={c['pilot_by_split']}")
    print(f"audit: group_overlap={a['group_overlap']} case_overlap={a['case_overlap']} "
          f"pilot_cases_in_test={a['pilot_cases_in_test']} pilot_cases_not_in_train={a['pilot_cases_not_in_train']} "
          f"digest={manifest['digest']}")
    print(f"wrote {paths.splits_parquet} and {digest_path}")
    return 0


def cmd_label_blind(ns: argparse.Namespace) -> int:
    from ucc_ml import dataset, labeling
    from ucc_ml.config import load_config

    cfg = load_config(ns.config)
    rp = labeling.round_paths(cfg, ns.round)
    if labeling.labels_exist(rp):
        print(f"REFUSED: {ns.round} already has labeller output under {rp.raw_dir}; its queue is frozen")
        return 1
    chunk_size = ns.chunk_size or cfg.labelling.chunk_size
    round_cases = dataset.read_frame(rp.cases)
    chunks, key = labeling.build_blind_queue(round_cases, ns.round, cfg.sampling.repeat_fraction, cfg.seed, chunk_size)
    for stale in sorted(rp.queue_dir.glob(f"queue_{ns.round}_part_*.csv")):
        stale.unlink()
    part_paths = []
    for part, chunk in enumerate(chunks, start=1):
        labeling.write_csv(chunk, rp.queue_part(part))
        part_paths.append(rp.queue_part(part))
    key_sha = labeling.write_csv(key, rp.key)
    labeling.write_preregistration(rp, part_paths, labeling.disclosure_for_round(cfg, ns.round))
    originals = key[~key.is_repeat]
    print(f"round={ns.round} cases={len(originals)} repeats={int(key.is_repeat.sum())} parts={len(chunks)} "
          f"chunk_size={chunk_size}")
    print(f"strata={originals.sampling_stratum.value_counts().sort_index().to_dict()}")
    for path in part_paths:
        print(f"wrote {path}")
    print(f"wrote {rp.key} sha256={key_sha} (private)")
    print(f"wrote {rp.preregistration}")
    return 0


def cmd_labeller_brief(ns: argparse.Namespace) -> int:
    from ucc_ml import labeling
    from ucc_ml.config import load_config
    from ucc_ml.provenance import write_json

    cfg = load_config(ns.config)
    rp = labeling.round_paths(cfg, ns.round)
    specs = cfg.path("specs_dir")
    policy_version = labeling.policy_version_for_round(cfg, ns.round)
    policy_text = (specs / f"{policy_version}.md").read_text(encoding="utf-8")
    if not labeling.policy_is_frozen(policy_text):
        print(f"REFUSED: {policy_version}.md is not 'status: FROZEN' (the Task 12 founder gate)")
        return 1
    parts = labeling.queue_parts(labeling.read_key(rp.key))
    if ns.part not in parts:
        print(f"REFUSED: {ns.round} has parts {parts}, not {ns.part}")
        return 1
    prompt_text = (specs / "labeller_prompt_v1.md").read_text(encoding="utf-8")
    chunk = labeling.read_queue(rp.queue_part(ns.part))
    brief_path = rp.brief(ns.part)
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    rp.structured_dir.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(labeling.render_labeller_brief(prompt_text, policy_text, chunk), encoding="utf-8")
    schema_path = rp.briefs_dir / "labeller_output_schema_v1.json"
    write_json(schema_path, labeling.LABELLER_OUTPUT_SCHEMA)
    print(f"brief {brief_path} rows={len(chunk)} part={ns.part} of {len(parts)} policy={policy_version}")
    print(f"schema {schema_path}")
    print("give the brief verbatim to one fresh tool-less subagent for pass a and to another for pass b")
    return 0


def cmd_labelling_status(ns: argparse.Namespace) -> int:
    from ucc_ml import labeling
    from ucc_ml.config import load_config

    cfg = load_config(ns.config)
    rp = labeling.round_paths(cfg, ns.round)
    parts = labeling.queue_parts(labeling.read_key(rp.key))
    status = labeling.labelling_status(rp, parts)
    for letter in labeling.PASSES:
        print(f"pass {letter}: {len(status[letter]['present'])}/{len(parts)} parts present; missing {status[letter]['missing']}")
    complete = not any(status[letter]["missing"] for letter in labeling.PASSES)
    print("COMPLETE" if complete else "INCOMPLETE")
    return 0 if complete else 1


def cmd_write_raw_labels(ns: argparse.Namespace) -> int:
    import json

    from ucc_ml import labeling
    from ucc_ml.config import load_config

    cfg = load_config(ns.config)
    rp = labeling.round_paths(cfg, ns.round)
    chunk = labeling.read_queue(rp.queue_part(ns.part))
    target = rp.raw_output(ns.pass_letter, ns.part)
    try:
        rows = labeling.rows_from_structured_output(json.loads(Path(ns.structured).read_text(encoding="utf-8")))
        sha = labeling.write_raw_output(rows, target, chunk.case_id.tolist(), cfg.labelling.reason_max_chars)
    except (ValueError, FileExistsError) as exc:
        print(f"REJECTED: {exc}")
        return 1
    print(f"wrote {target} rows={len(rows)} labels={rows.label.value_counts().sort_index().to_dict()} sha256={sha}")
    return 0


def cmd_import_labels(ns: argparse.Namespace) -> int:
    from ucc_ml import labeling
    from ucc_ml.config import load_config

    cfg = load_config(ns.config)
    rp = labeling.round_paths(cfg, ns.round)
    try:
        passes = labeling.import_round_passes(rp, cfg.labelling.reason_max_chars)
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 1
    for letter, frame in passes.items():
        sha = labeling.write_csv(frame, rp.pass_file(letter))
        originals = frame[~frame.is_repeat]
        print(f"pass {letter}: cases={len(originals)} repeats={int(frame.is_repeat.sum())} "
              f"labels={originals.label.value_counts().sort_index().to_dict()} sha256={sha}")
    labeling.write_passes_digest(rp, labeling.queue_parts(labeling.read_key(rp.key)),
                                 labeling.disclosure_for_round(cfg, ns.round))
    print(f"wrote {rp.pass_file('a')}, {rp.pass_file('b')} and {rp.passes_digest}")
    return 0


def cmd_review_workbook(ns: argparse.Namespace) -> int:
    from ucc_ml import dataset, labeling
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.provenance import utc_now_iso, write_json
    from ucc_ml.splitting import read_splits

    cfg = load_config(ns.config)
    rp = labeling.round_paths(cfg, ns.round)
    if rp.workbook.exists():
        print(f"REFUSED: {rp.workbook} already exists; it may hold founder decisions and is never regenerated")
        return 1
    pass_a, pass_b = labeling.read_pass_file(rp.pass_file("a")), labeling.read_pass_file(rp.pass_file("b"))
    report = labeling.agreement_report(pass_a, pass_b, ns.round)
    write_json(rp.agreement, report)
    splits = read_splits(artefact_paths(cfg).splits_parquet)
    per_cell = cfg.labelling.founder_audit_per_split_stratum
    review = labeling.select_founder_review(pass_a, pass_b, splits, per_cell, cfg.seed, ns.round)
    sha = labeling.build_founder_workbook(review, pass_a, pass_b, dataset.read_frame(rp.cases), rp.workbook)
    audit = review[review.review_type == "audit"]
    write_json(rp.review_manifest, {
        "round": ns.round, "workbook": rp.workbook.name, "workbook_sha256_issued": sha, "created_at": utc_now_iso(),
        "seed": cfg.seed, "founder_audit_per_split_stratum": per_cell, "rows": int(len(review)),
        "disagreements": sorted(review.case_id[review.review_type == "disagreement"]),
        "audit": sorted(audit.case_id),
        "audit_by_split_stratum": {split: {stratum: int(n) for stratum, n in group.sampling_stratum.value_counts().sort_index().items()}
                                   for split, group in audit.groupby("split", sort=True)},
    })
    agreement, consistency = report["pass_agreement"], report["repeat_consistency"]
    print(f"pass agreement: {agreement['agreed']}/{agreement['n']} rate={agreement['rate']}")
    print(f"repeat consistency: pass_a {consistency['pass_a']['consistent']}/{consistency['pass_a']['n']}, "
          f"pass_b {consistency['pass_b']['consistent']}/{consistency['pass_b']['n']}")
    print(f"founder review rows={len(review)} disagreements={len(report['disagreements'])} audit={len(audit)}")
    print(f"wrote {rp.agreement}, {rp.workbook} and {rp.review_manifest}")
    return 0


def cmd_import_founder_review(ns: argparse.Namespace) -> int:
    from ucc_ml import labeling
    from ucc_ml.config import load_config
    from ucc_ml.provenance import read_json

    cfg = load_config(ns.config)
    rp = labeling.round_paths(cfg, ns.round)
    manifest = read_json(rp.review_manifest)
    pass_a, pass_b = labeling.read_pass_file(rp.pass_file("a")), labeling.read_pass_file(rp.pass_file("b"))
    try:
        rows = labeling.read_founder_workbook(rp.workbook)
        decisions, summary = labeling.import_founder_review(rows, manifest, pass_a, pass_b,
                                                            labeling.file_mtime_iso(rp.workbook))
    except ValueError as exc:
        print(f"REJECTED: {exc}")
        return 1
    labeling.write_csv(decisions, rp.founder_decisions)
    labeling.write_founder_digest(rp, manifest["workbook_sha256_issued"])
    disagreements, audit = summary["disagreements"], summary["audit"]
    print(f"disagreements: {disagreements['decided']}/{disagreements['n']} decided")
    print(f"audit: {audit['n_audited']}/{audit['n_selected']} reviewed, confirmed={audit['n_confirmed']} "
          f"overturned={audit['n_overturned']} agreement_rate={audit['agreement_rate']}")
    print(f"wrote {rp.founder_decisions} and {rp.founder_digest}")
    if disagreements["undecided"]:
        print(f"INCOMPLETE: {disagreements['undecided']} disagreement(s) still have no founder decision; "
              "validate-labels will refuse")
        return 1
    return 0


def cmd_validate_labels(ns: argparse.Namespace) -> int:
    from ucc_ml import labeling
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.provenance import git_head, sha256_file, utc_now_iso, write_json

    cfg = load_config(ns.config)
    paths = artefact_paths(cfg)
    rounds = [r for r in labeling.LABELLING_ROUNDS
              if any(labeling.round_paths(cfg, r).pass_file(letter).exists() for letter in labeling.PASSES)]
    allowed = cfg.labelling.rounds_in_labels
    if allowed:
        skipped = [r for r in rounds if r not in allowed]
        rounds = [r for r in rounds if r in allowed]
        if skipped:
            print(f"not a label source (labelling.rounds_in_labels): {', '.join(skipped)}")
    if "pilot_v1" not in rounds:
        print("REFUSED: the pilot_v1 passes have not been imported (run import-labels --round pilot_v1)")
        return 1
    frames, agreements, founder, inputs = {}, {}, {}, {}
    for round_name in rounds:
        rp = labeling.round_paths(cfg, round_name)
        try:
            frames[round_name], agreements[round_name], founder[round_name] = labeling.load_round_for_validation(
                rp, labeling.policy_version_for_round(cfg, round_name),
                labeling.disagreement_policy_for_round(cfg, round_name))
        except (labeling.UndecidedDisagreements, FileNotFoundError) as exc:
            print(f"REFUSED: {exc}; labels.csv was not written")
            return 1
        for used in (rp.pass_file("a"), rp.pass_file("b"), rp.review_manifest, rp.founder_decisions):
            if used.exists():   # a round under the "unresolved" policy has no founder decisions file
                inputs[str(used.relative_to(cfg.repo_root))] = sha256_file(used)
    labels = labeling.build_labels(frames)
    labels_sha = labeling.write_csv(labels, paths.labels_csv)
    disclosure_by_round = {r: labeling.disclosure_for_round(cfg, r) for r in rounds}
    manifest = labeling.labels_manifest(
        labels, labels_sha, agreements, founder, cfg.version.label_policy_version,
        extra={"created_at": utc_now_iso(), "git_head": git_head(cfg.repo_root), "config_sha256": cfg.config_sha256,
               "inputs_sha256": inputs,
               "policy_version_by_round": {r: labeling.policy_version_for_round(cfg, r) for r in rounds},
               "disagreement_policy_by_round": {r: labeling.disagreement_policy_for_round(cfg, r) for r in rounds},
               "disclosure_by_round": disclosure_by_round,
               # overrides the module default: this file holds more than one arrangement
               "disclosure": labeling.pooled_disclosure(disclosure_by_round)})
    write_json(paths.labels_manifest, manifest)
    for round_name in rounds:
        rp = labeling.round_paths(cfg, round_name)
        report = labeling.round_report(labels, agreements[round_name], founder[round_name], round_name)
        write_json(rp.report, report)
        shares = {s: v["insufficient_share"] for s, v in report["labelability"].items()}
        prevalence = {s: v["relevant_share"] for s, v in report["relevant_prevalence"].items()}
        print(f"{round_name}: cases={report['cases']} counts_by_status={report['counts_by_status']}")
        print(f"  INSUFFICIENT_EVIDENCE share by stratum={shares}")
        print(f"  RELEVANT share by stratum={prevalence}")
        print(f"  pass agreement={report['pass_agreement']['agreed']}/{report['pass_agreement']['n']} "
              f"founder audit={report['founder_audit']['n_confirmed']}/{report['founder_audit']['n_audited']} "
              f"repeat consistency a={report['repeat_consistency']['pass_a']['consistent']}/{report['repeat_consistency']['pass_a']['n']} "
              f"b={report['repeat_consistency']['pass_b']['consistent']}/{report['repeat_consistency']['pass_b']['n']}")
        print(f"  wrote {rp.report}")
    digest = labeling.write_labels_digest(paths.public_data_dir, paths.labels_csv, paths.labels_manifest, manifest)
    print(f"labels rows={manifest['rows']} counts_by_status={manifest['counts_by_status']} "
          f"counts_by_round={manifest['counts_by_round']} ({manifest['disclosure']})")
    print(f"wrote {paths.labels_csv}, {paths.labels_manifest} and {digest}")
    return 0


def cmd_build_screen(ns: argparse.Namespace) -> int:
    from ucc_ml.screening import build_screen

    out = build_screen(ns.config)
    print(out["counts"].to_string(index=False))
    m = out["manifest"]
    print(f"fit on {m['fit_split']}: {m['fit_positives']:,} positive / {m['fit_negatives']:,} negative")
    print(f"rules words among the top 200 features: {m['rules_words_in_top_200_features']}")
    print(f"wrote {out['cells']} sha256={m['screen_cells_sha256']}")
    return 0


def cmd_make_main_round(ns: argparse.Namespace) -> int:
    from ucc_ml.config import artefact_paths, load_config
    from ucc_ml.dataset import read_candidates, read_frame
    from ucc_ml.provenance import git_head, sha256_file, utc_now_iso, write_json
    from ucc_ml.sampling import draw_main_round, propose_main_allocation, split_cell_populations
    from ucc_ml.splitting import read_splits

    cfg = load_config(ns.config)
    paths = artefact_paths(cfg)
    cases = read_candidates(paths.candidates_parquet)
    splits = read_splits(paths.splits_parquet)
    cells = read_frame(paths.screen_cells)
    budget = cfg.section("budget")["main_round"]
    totals = {k: int(budget[k]) for k in ("train", "validation", "test")}
    floor = int(cfg.section("screening")["b4_floor_per_split"])
    pops = split_cell_populations(cases, splits, cells)
    alloc = propose_main_allocation(pops, totals, b4_floor_per_split=floor)

    for split in ("train", "validation", "test"):
        print(f"{split}: {sum(alloc[split].values())} cases over {len(alloc[split])} cells; "
              f"unscreened floor {sum(n for (_, c), n in alloc[split].items() if c == 'B4_remainder')} >= {floor}")
    if ns.dry_run:
        return 0

    pilot = read_frame(paths.pilot_cases).case_id.tolist()
    drawn = draw_main_round(cases, splits, cells, alloc, seed=cfg.seed, exclude_case_ids=pilot)
    paths.main_cases.parent.mkdir(parents=True, exist_ok=True)
    drawn.to_parquet(paths.main_cases, index=False)
    write_json(paths.main_manifest, {
        "purpose": "main_v1",
        "policy_version": cfg.labelling.policy_version_by_round.get("main_v1"),
        "rows": int(len(drawn)),
        "totals": totals,
        "b4_floor_per_split": floor,
        "allocation": {s: {f"{k[0]}|{k[1]}": v for k, v in a.items()} for s, a in alloc.items()},
        "populations": {s: {f"{k[0]}|{k[1]}": v for k, v in a.items()} for s, a in pops.items()},
        "by_split": {k: int(v) for k, v in drawn.split.value_counts().items()},
        "by_cell": {k: int(v) for k, v in drawn.screen_cell.value_counts().items()},
        "excluded_pilot_cases": len(pilot),
        "seed": cfg.seed,
        "candidates_sha256": sha256_file(paths.candidates_parquet),
        "splits_sha256": sha256_file(paths.splits_parquet),
        "screen_cells_sha256": sha256_file(paths.screen_cells),
        "main_cases_sha256": sha256_file(paths.main_cases),
        "created_at": utc_now_iso(), "git_head": git_head(cfg.repo_root),
        "config_sha256": cfg.config_sha256,
    })
    print(f"wrote {paths.main_cases} rows={len(drawn)} sha256={sha256_file(paths.main_cases)}")
    print(f"wrote {paths.main_manifest}")
    return 0


# --- cmd_<name> functions are added above this line by later tasks; each imports its implementation lazily ---


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.command is None:
        parser.print_help()
        return 2
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
