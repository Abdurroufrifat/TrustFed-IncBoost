from __future__ import annotations

from pathlib import Path

import yaml


def test_unseen_confirmation_protocol_is_frozen_and_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    matrix = yaml.safe_load(
        (root / "configs" / "ehms_unseen_confirmation.yaml").read_text(
            encoding="utf-8"
        )
    )
    seeds = {int(value) for value in matrix["seeds"]}
    development = {
        int(value) for value in matrix["protocol"]["development_seeds"]
    }
    assert len(seeds) == 10
    assert seeds.isdisjoint(development)
    assert matrix["protocol"]["planned_runs"] == 90
    assert matrix["protocol"]["no_post_result_tuning_permitted"] is True

    profiles = matrix["aggregation_profiles"]
    assert profiles["uniform"]["method"] == "uniform"
    assert profiles["trust_v1"]["method"] == "trust_aware"
    assert profiles["trust_v1"]["consensus_strength"] == 0.25
    assert profiles["trust_v2"]["method"] == "trust_aware_v2"
    assert profiles["trust_v2"]["consensus_strength"] == 0.10

    cells = {
        (condition["scenario"], condition["profile"])
        for condition in matrix["conditions"]
    }
    assert cells == {
        (scenario, profile)
        for scenario in ("clean", "largest_flip", "attack_rich_flip")
        for profile in ("uniform", "trust_v1", "trust_v2")
    }
