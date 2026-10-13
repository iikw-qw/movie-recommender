import numpy as np
import pytest

from src import core, data


@pytest.fixture(scope="module")
def prepared():
    ratings = data.generate_ratings(n_users=400, n_items=120, density=0.08, seed=1)
    n_users, n_items = ratings["user"].max() + 1, ratings["item"].max() + 1
    train, test = core.train_test_split_ratings(ratings)
    return ratings, train, test, n_users, n_items


def test_split_has_no_overlap(prepared):
    _, train, test, _, _ = prepared
    a = set(zip(train["user"], train["item"]))
    b = set(zip(test["user"], test["item"]))
    assert not a & b


def test_split_sizes_add_up(prepared):
    ratings, train, test, _, _ = prepared
    assert len(train) + len(test) == len(ratings)


def test_bias_model_beats_global_mean(prepared):
    _, train, test, nu, ni = prepared
    model = core.BiasModel().fit(train, nu, ni)
    baseline = np.sqrt(np.mean((test["rating"] - train["rating"].mean()) ** 2))
    assert core.rmse(model, test) < baseline


def test_als_improves_on_bias_model(prepared):
    _, train, test, nu, ni = prepared
    bias = core.BiasModel().fit(train, nu, ni)
    als = core.ALS(factors=6, iters=8).fit(train, nu, ni)
    assert core.rmse(als, test) < core.rmse(bias, test)


def test_ranking_metrics_are_probabilities(prepared):
    _, train, test, nu, ni = prepared
    model = core.Popularity().fit(train, nu, ni)
    m = core.ranking_metrics(model, train, test, k=10)
    for key in ("precision@k", "recall@k", "ndcg@k"):
        assert 0 <= m[key] <= 1


def test_seen_items_are_never_recommended(prepared):
    _, train, test, nu, ni = prepared
    model = core.BiasModel().fit(train, nu, ni)
    user = int(train["user"].iloc[0])
    seen = set(train.loc[train["user"] == user, "item"])
    scores = model.score_all(user).copy()
    scores[list(seen)] = -np.inf
    assert not (set(np.argsort(-scores)[:10]) & seen)
