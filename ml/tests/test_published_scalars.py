"""Two things that reach PUBLIC documents and were being published as single values.

Both defects are the same shape: a quantity that is really per round, or really a pair computed under
different assumptions, flattened to one number that is then true of only part of what it covers.

  * `labels_policy_version` is written into model-manifest.json, which is the FIRST of the four
    manifests hashed into `release_id`. A false value there is baked into a permanent public
    identifier that cannot be corrected without changing the id and invalidating every link to it.
  * the model card prints a point estimate beside an interval. The estimate is the design-weighted
    plug-in and the interval is smoothed by a Jeffreys prior, so at the boundary they disagree. The
    flag was computed and carried in metrics.json all along; the card just never read it, so the only
    document a reader actually sees showed an incoherent pair in silence.
"""
from __future__ import annotations

from ucc_ml.inference import _interval
from ucc_ml.labeling import pooled_policy_version


def test_one_policy_across_the_rounds_collapses_to_that_policy():
    assert pooled_policy_version({"pilot_v1": "label_policy_v1", "main_v1": "label_policy_v1"},
                                 "ignored") == "label_policy_v1"


def test_rounds_under_different_policies_are_each_named():
    """The real labels: 240 pilot rows under v1, 2,880 main rows under v2. Naming either one alone
    states the smaller round's policy as if it covered the file."""
    assert pooled_policy_version({"main_v1": "label_policy_v2", "pilot_v1": "label_policy_v1"},
                                 "label_policy_v1") == (
        "mixed by round -- main_v1: label_policy_v2; pilot_v1: label_policy_v1")


def test_no_map_falls_back_to_the_scalar():
    """A manifest stating no map is itself the claim that one policy covers every round."""
    assert pooled_policy_version(None, "label_policy_v1") == "label_policy_v1"
    assert pooled_policy_version({}, "label_policy_v1") == "label_policy_v1"


def test_the_card_marks_an_estimate_that_falls_outside_its_own_interval():
    outside = {"estimate": 1.0, "lower": 0.544, "upper": 0.997, "estimate_outside_interval": True}
    assert _interval(outside) == "1.000 [0.544, 0.997] (estimate outside interval)"


def test_the_card_leaves_a_coherent_pair_unmarked():
    inside = {"estimate": 0.82, "lower": 0.71, "upper": 0.93, "estimate_outside_interval": False}
    assert _interval(inside) == "0.820 [0.710, 0.930]"
    # a block predating the flag must not acquire a mark it never earned
    assert _interval({"estimate": 0.82, "lower": 0.71, "upper": 0.93}) == "0.820 [0.710, 0.930]"
