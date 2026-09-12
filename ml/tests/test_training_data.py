"""Plan A's artefacts through Plan A's readers, the K3 label resolution, the model table in the K2 colon form,
stratum alignment, the TRAIN design weights and the K3 label statistics block."""
import numpy as np
import pandas as pd
import pytest

from ucc_ml.training import (
    MODEL_TABLE_COLUMNS,
    build_model_table,
    check_stratum_alignment,
    labels_summary,
    read_plan_a_inputs,
    resolve_binary_labels,
    resolved,
    split_populations,
    training_weights,
)


@pytest.fixture(scope="module")
def world_and_paths(tmp_path_factory):
    from ucc_ml.synthetic import make_world, write_world

    world = make_world()
    return world, write_world(world, tmp_path_factory.mktemp("world"))


def test_plan_a_readers_return_the_world(world_and_paths):
    world, paths = world_and_paths
    inputs = read_plan_a_inputs(paths)
    assert len(inputs.candidates) == len(world.candidates) and len(inputs.labels) == len(world.labels)
    assert inputs.labels.is_repeat.dtype == bool and isinstance(inputs.candidates.lender_names_raw.iloc[0], list)
    assert inputs.labels_manifest["disclosure"] == "model-labelled, founder-adjudicated"


def test_resolve_binary_labels_keeps_the_three_resolved_statuses_and_drops_blind_repeats():
    labels = pd.DataFrame({
        "case_id": [f"{i:064x}" for i in range(6)],
        "label": ["RELEVANT", "NOT_RELEVANT", "INSUFFICIENT_EVIDENCE", "RELEVANT", "NOT_RELEVANT", "RELEVANT"],
        "adjudication_status": ["model_agreed", "founder_confirmed", "founder_adjudicated", "model_agreed", "blind_repeat",
                                "founder_adjudicated"],
        "is_repeat": [False, False, False, False, True, False]})
    res = resolve_binary_labels(labels)
    assert len(res) == 5 and "blind_repeat" not in set(res.adjudication_status)
    assert res.y.tolist()[:2] == [1.0, 0.0] and np.isnan(res.y.iloc[2]) and res.y.tolist()[3:] == [1.0, 1.0]


@pytest.mark.parametrize("status", ["adjudicated", "pending", "MODEL_AGREED"])
def test_an_unknown_status_raises(world_and_paths, status):
    world, _ = world_and_paths
    bad = world.labels.copy()
    bad.loc[bad.index[0], "adjudication_status"] = status
    with pytest.raises(ValueError, match=status):
        resolve_binary_labels(bad)


def test_resolve_binary_labels_refuses_conflicting_labels(world_and_paths):
    world, _ = world_and_paths
    first = world.labels[~world.labels.is_repeat].head(1).copy()
    first["label"] = np.where(first.label == "RELEVANT", "NOT_RELEVANT", "RELEVANT")
    with pytest.raises(ValueError, match="conflicting"):
        resolve_binary_labels(pd.concat([world.labels, first], ignore_index=True))


def test_build_model_table_joins_and_leaves_weights_to_the_split(world_and_paths):
    from ucc_ml.contracts import STRATA, stratum_name

    world, _ = world_and_paths
    table = build_model_table(world.candidates, world.splits, world.labels)
    originals = world.labels[~world.labels.is_repeat]
    assert list(table.columns) == list(MODEL_TABLE_COLUMNS) and len(table) == len(originals)
    assert set(table.stratum) == set(STRATA)
    assert (table.stratum == [stratum_name(r, b) for r, b in zip(table.region, table.baseline_qualifies)]).all()
    assert table.sample_weight.isna().all()
    assert set(table.labelling_round) == {"pilot_v1", "main_v1"}
    check_stratum_alignment(table)
    assert resolved(table).y.notna().all() and len(resolved(table)) < len(table)


def test_build_model_table_refuses_unknown_case_ids(world_and_paths):
    world, _ = world_and_paths
    extra = world.labels.head(1).copy()
    extra["case_id"] = "f" * 64
    with pytest.raises(ValueError, match="f" * 64):
        build_model_table(world.candidates, world.splits, pd.concat([world.labels, extra], ignore_index=True))


def test_check_stratum_alignment_names_a_mislabelled_stratum(world_and_paths):
    world, _ = world_and_paths
    table = build_model_table(world.candidates, world.splits, world.labels)
    rows = table.index[table.sampling_stratum == "CO:accepted"][:3]
    table.loc[rows, "sampling_stratum"] = "CO:rejected"
    with pytest.raises(ValueError, match=table.case_id[rows[0]]):
        check_stratum_alignment(table)


def test_split_populations_count_every_candidate(world_and_paths):
    from ucc_ml.contracts import stratum_name

    world, _ = world_and_paths
    pops = split_populations(world.candidates, world.splits)
    joined = world.candidates.merge(world.splits[["case_id", "split"]], on="case_id")
    manual = joined.groupby(["split", [stratum_name(r, b) for r, b in zip(joined.region, joined.baseline_qualifies)]]).size()
    assert sum(sum(v.values()) for v in pops.values()) == len(world.candidates)
    assert all(pops[split][stratum] == n for (split, stratum), n in manual.items())


def test_training_weights_are_inverse_inclusion_probability_and_mean_one():
    """The protocol pins this: each TRAIN row carries the rate of the CELL it was drawn from.

    Two rows of ONE stratum drawn at different rates must therefore NOT share a weight. N_h / n_h
    assumed they did, which was true only while a stratum held one cell at one rate."""
    table = pd.DataFrame({"split": ["train"] * 6 + ["test"],
                          "stratum": ["CO:accepted"] * 3 + ["CO:rejected"] * 3 + ["CO:rejected"],
                          "inclusion_probability": [0.05, 0.05, 0.05, 0.01, 0.002, 0.002, 0.5],
                          "y": [1, 0, np.nan, 0, 0, 1, 1]})
    w = training_weights(table).to_numpy()
    assert len(w) == 6                                             # TRAIN rows only
    raw = np.array([20.0, 20.0, 20.0, 100.0, 500.0, 500.0])        # 1 / inclusion_probability
    fitted = np.array([True, True, False, True, True, True])       # y is NaN on row 2
    assert w == pytest.approx(raw / raw[fitted].mean())
    assert w[fitted].mean() == pytest.approx(1.0)
    assert w[3] != pytest.approx(w[4])   # same stratum, different rate -> different weight


def test_pilot_and_main_rows_of_one_stratum_keep_their_own_rates(world_and_paths):
    """A stratum must NOT collapse to a single weight: the pilot and the main round drew it at
    different rates, and collapsing them is precisely the bias N_h / n_h introduced."""
    world, _ = world_and_paths
    table = build_model_table(world.candidates, world.splits, world.labels)
    train = table[table.split == "train"].copy()
    train["w"] = training_weights(table).to_numpy()
    assert (train.groupby("stratum").w.nunique() > 1).any()
    inverse_pi = train.groupby(["stratum", "labelling_round"]).inclusion_probability.first().rdiv(1.0).unstack()
    assert ((inverse_pi["pilot_v1"] / inverse_pi["main_v1"]) > 1.5).all()   # pilot drew at a lower rate
    product = (train.w * train.inclusion_probability).to_numpy()            # w = c / pi exactly
    assert np.allclose(product, product[0])


def test_labels_summary_carries_the_k3_statistics(world_and_paths):
    world, _ = world_and_paths
    table = build_model_table(world.candidates, world.splits, world.labels)
    summary = labels_summary(table, "validation", world.labels_manifest)
    for key in ("disclosure", "counts_by_status", "counts_by_round", "pass_agreement", "founder_audit", "repeat_consistency"):
        assert summary[key] == world.labels_manifest[key]
    sub = table[table.split == "validation"]
    assert summary["n_labelled"] == len(sub) and summary["n_unresolved"] == int(sub.y.isna().sum())
    with pytest.raises(ValueError, match="disclosure"):
        labels_summary(table, "validation", {**world.labels_manifest, "disclosure": "human-labelled"})
    with pytest.raises(ValueError, match="founder_audit"):
        labels_summary(table, "validation", {k: v for k, v in world.labels_manifest.items() if k != "founder_audit"})


def _manifest(**over):
    base = {"disclosure": "model-labelled, founder-adjudicated", "counts_by_status": {}, "counts_by_round": {},
            "pass_agreement": {}, "founder_audit": {}, "repeat_consistency": {}, "policy_version": "label_policy_v1"}
    return {**base, **over}


def _one_row_table():
    return pd.DataFrame({"split": ["train"], "y": [1.0]})


def test_labels_summary_accepts_a_disclosure_that_names_each_round():
    """A file may hold rounds labelled under different arrangements, and it must say so. Refusing
    anything but the single literal stops every metrics document the moment one round stops being
    founder-adjudicated; accepting any string lets a false provenance claim through."""
    from ucc_ml.labeling import pooled_disclosure

    by_round = {"pilot_v1": "model-labelled, founder-adjudicated",
                "main_v1": "model-labelled, two independent blind passes, disagreements retained as unresolved"}
    m = _manifest(disclosure=pooled_disclosure(by_round), disclosure_by_round=by_round)
    block = labels_summary(_one_row_table(), "train", m)
    assert block["disclosure"] == pooled_disclosure(by_round)
    assert block["disclosure_by_round"] == by_round      # carried into the metrics document


def test_labels_summary_refuses_a_disclosure_that_contradicts_its_own_rounds():
    by_round = {"pilot_v1": "model-labelled, founder-adjudicated",
                "main_v1": "model-labelled, two independent blind passes, disagreements retained as unresolved"}
    m = _manifest(disclosure="model-labelled, founder-adjudicated", disclosure_by_round=by_round)
    with pytest.raises(ValueError, match="does not describe its own"):
        labels_summary(_one_row_table(), "train", m)


def test_labels_summary_still_requires_the_literal_when_there_is_no_per_round_map():
    with pytest.raises(ValueError, match="disclosure must be"):
        labels_summary(_one_row_table(), "train", _manifest(disclosure="something else"))


def test_labels_summary_names_the_policy_of_each_round_when_they_differ():
    """'policy_version' had exactly the defect 'disclosure' had. The scalar is the config default and
    names the PILOT's policy; on the real file main_v1 is 2880 of 3120 rows under label_policy_v2, so
    copying the scalar states the smaller round's policy as if it covered the whole file."""
    by_round = {"pilot_v1": "label_policy_v1", "main_v1": "label_policy_v2"}
    block = labels_summary(_one_row_table(), "train",
                           _manifest(policy_version="label_policy_v1", policy_version_by_round=by_round))
    assert block["policy_version"] == "mixed by round -- main_v1: label_policy_v2; pilot_v1: label_policy_v1"
    assert block["policy_version_by_round"] == by_round     # carried into the metrics document
    assert block["policy_version"] != "label_policy_v1"     # the majority round is no longer misnamed


def test_labels_summary_keeps_the_scalar_when_every_round_shares_one_policy():
    """A single-valued map says nothing the scalar does not, so the scalar still stands."""
    by_round = {"pilot_v1": "label_policy_v1", "main_v1": "label_policy_v1"}
    block = labels_summary(_one_row_table(), "train", _manifest(policy_version_by_round=by_round))
    assert block["policy_version"] == "label_policy_v1"
    assert block["policy_version_by_round"] == by_round


def test_labels_summary_uses_the_scalar_when_there_is_no_per_round_policy_map():
    block = labels_summary(_one_row_table(), "train", _manifest())
    assert block["policy_version"] == "label_policy_v1"
    assert "policy_version_by_round" not in block


def test_labels_summary_prefers_the_map_over_a_stale_scalar_without_raising():
    """Unlike the disclosure, a policy scalar that disagrees with the map is NOT an error: cli.py
    writes the config default there beside the map, so raising would refuse the pipeline's own
    output. The map is the truth and wins."""
    block = labels_summary(_one_row_table(), "train",
                           _manifest(policy_version="label_policy_v1",
                                     policy_version_by_round={"main_v1": "label_policy_v2"}))
    assert block["policy_version"] == "label_policy_v2"
