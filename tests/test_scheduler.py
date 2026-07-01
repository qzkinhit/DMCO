from dmco.scheduler import CostAwareUCB


def test_cost_aware_ucb_tracks_reward_per_cost():
    mab = CostAwareUCB(last_metric=0.5)
    arm = mab.choose(["clean", "automl"])
    assert arm == "clean"
    reward = mab.observe("clean", metric=0.6, cost=2.0)
    assert reward > 0
    assert mab.counts["clean"] > 0
    assert mab.best_metric == 0.6


def test_cost_aware_ucb_prioritizes_untried_arms():
    mab = CostAwareUCB(last_metric=0.0)
    assert mab.choose(["clean", "automl"]) == "clean"
    mab.observe("clean", 0.1, 1.0)
    assert mab.choose(["clean", "automl"]) == "automl"
