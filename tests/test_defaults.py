from utils.defaults import max_dist, max_rank, feature_count


def test_defaults_match_config():
    from config.base_config import HIGHWAY_CONFIG as C

    # Check max_dist matches config's feature_range bounds
    assert max_dist() == max(
        abs(C["observation"]["features_range"]["x"][0]),
        abs(C["observation"]["features_range"]["x"][1]),
        abs(C["observation"]["features_range"]["y"][0]),
        abs(C["observation"]["features_range"]["y"][1]),
    )
    # Check max_rank matches vehicles_count
    assert max_rank() == C["observation"]["vehicles_count"]
    # Check feature_count matches length of features list
    assert feature_count() == len(C["observation"]["features"])
