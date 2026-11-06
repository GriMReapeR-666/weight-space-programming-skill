"""Tests for Introspective Simulation, Critic Discernment, and TPT Sampler."""

from engine import (
    CriticDiscernmentEngine,
    ErrorCatalogManager,
    FailurePattern,
    ReasoningRollout,
    ThinkPruneTrainEngine,
)


def test_error_catalog_defaults_and_match():
    catalog = ErrorCatalogManager()
    assert len(catalog.patterns) >= 4
    assert "WSP-ERR-001" in catalog.patterns

    matched = catalog.find_matching_pattern("The model enters an attention copying loop indefinitely.")
    assert matched is not None
    assert matched.pattern_id == "WSP-ERR-001"


def test_error_catalog_register_new():
    catalog = ErrorCatalogManager()
    new_pat = FailurePattern(
        pattern_id="WSP-ERR-999",
        name="Custom Routing Divergence",
        description="Router assigns all inputs to single expert.",
        typical_symptom="Expert 0 receives 100% tokens.",
        recommended_intervention="Entropy regularization on gating logits.",
    )
    catalog.register_pattern(new_pat)
    assert "WSP-ERR-999" in catalog.patterns

    exported = catalog.export_catalog()
    assert any(p["pattern_id"] == "WSP-ERR-999" for p in exported)


def test_critic_discernment_game_evaluation():
    catalog = ErrorCatalogManager()
    pair = CriticDiscernmentEngine.generate_critique_pair(
        trace_id="trace_002",
        problem_description="Solve multi-step derivation",
        reasoning_trace="Step 1 ok, step 2 ok, step 3 dropped variable x.",
        catalog=catalog,
    )

    correct_choice = "A" if pair.helpful_is_option_a else "B"
    wrong_choice = "B" if pair.helpful_is_option_a else "A"

    is_correct, reward = CriticDiscernmentEngine.evaluate_discernment(pair, correct_choice)
    assert is_correct
    assert reward == 1.0

    is_wrong, penalty = CriticDiscernmentEngine.evaluate_discernment(pair, wrong_choice)
    assert not is_wrong
    assert penalty == -0.5


def test_think_prune_train_filtering():
    rollouts = [
        ReasoningRollout(rollout_id="r1", problem="2 + 2", reasoning_trace="2 + 2 is 4", candidate_answer="4"),
        ReasoningRollout(rollout_id="r2", problem="2 + 2", reasoning_trace="2 + 2 is 5", candidate_answer="5"),
        ReasoningRollout(rollout_id="r3", problem="3 * 3", reasoning_trace="3 * 3 is 9", candidate_answer="9"),
    ]

    def mock_verifier(prob: str, ans: str) -> bool:
        return (prob == "2 + 2" and ans == "4") or (prob == "3 * 3" and ans == "9")

    batch = ThinkPruneTrainEngine.prune_and_filter(rollouts, mock_verifier)
    assert batch.total_rollouts == 3
    assert batch.pruned_correct_count == 2
    assert len(batch.retained_traces) == 2
    assert len(batch.rejected_traces) == 1
    assert abs(batch.success_rate - (2 / 3)) < 1e-4

    sft_data = ThinkPruneTrainEngine.export_sft_dataset(batch)
    assert len(sft_data) == 2
    assert "<think>" in sft_data[0]["output"]
    assert "4" in sft_data[0]["output"]
