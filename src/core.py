import numpy as np
import pandas as pd
from scipy import sparse


def train_test_split_ratings(ratings, test_per_user=3, seed=0):
    """Hold out a few ratings of every user who has enough of them."""
    rng = np.random.default_rng(seed)
    counts = ratings.groupby("user")["item"].transform("size")
    eligible = ratings[counts > test_per_user + 2]
    test_idx = (
        eligible.groupby("user", group_keys=False)
        .apply(lambda g: g.loc[rng.choice(g.index, test_per_user, replace=False)])
        .index
    )
    test = ratings.loc[test_idx]
    train = ratings.drop(test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


class Popularity:
    """Rank items by their damped average rating; ignores the user."""

    def fit(self, train, n_users, n_items, damping=5.0):
        self.mean = train["rating"].mean()
        g = train.groupby("item")["rating"].agg(["sum", "count"])
        score = np.full(n_items, self.mean)
        score[g.index] = (g["sum"] + damping * self.mean) / (g["count"] + damping)
        self.item_score = score
        self.n_users = n_users
        return self

    def predict(self, users, items):
        return self.item_score[np.asarray(items)]

    def score_all(self, user):
        return self.item_score


class BiasModel:
    """Global mean plus user and item offsets, fitted by alternating shrunken averages."""

    def fit(self, train, n_users, n_items, reg=8.0, iters=10):
        self.mu = train["rating"].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        u, i, r = train["user"].to_numpy(), train["item"].to_numpy(), train["rating"].to_numpy()
        for _ in range(iters):
            res = r - self.mu - self.bu[u]
            self.bi = np.bincount(i, res, n_items) / (np.bincount(i, minlength=n_items) + reg)
            res = r - self.mu - self.bi[i]
            self.bu = np.bincount(u, res, n_users) / (np.bincount(u, minlength=n_users) + reg)
        return self

    def predict(self, users, items):
        return self.mu + self.bu[np.asarray(users)] + self.bi[np.asarray(items)]

    def score_all(self, user):
        return self.mu + self.bu[user] + self.bi


class ALS:
    """Matrix factorisation on the residuals of the bias model, fitted with alternating least squares."""

    def __init__(self, factors=8, reg=6.0, iters=12, seed=0):
        self.k, self.reg, self.iters, self.seed = factors, reg, iters, seed

    def fit(self, train, n_users, n_items):
        self.base = BiasModel().fit(train, n_users, n_items)
        u, i = train["user"].to_numpy(), train["item"].to_numpy()
        resid = train["rating"].to_numpy() - self.base.predict(u, i)
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_items, self.k))
        by_user = [np.where(u == x)[0] for x in range(n_users)]
        by_item = [np.where(i == x)[0] for x in range(n_items)]
        eye = self.reg * np.eye(self.k)
        for _ in range(self.iters):
            for x in range(n_users):
                idx = by_user[x]
                if len(idx):
                    q = self.Q[i[idx]]
                    self.P[x] = np.linalg.solve(q.T @ q + eye, q.T @ resid[idx])
            for y in range(n_items):
                idx = by_item[y]
                if len(idx):
                    p = self.P[u[idx]]
                    self.Q[y] = np.linalg.solve(p.T @ p + eye, p.T @ resid[idx])
        return self

    def predict(self, users, items):
        users, items = np.asarray(users), np.asarray(items)
        return self.base.predict(users, items) + np.einsum("ij,ij->i", self.P[users], self.Q[items])

    def score_all(self, user):
        return self.base.score_all(user) + self.Q @ self.P[user]


def rmse(model, test):
    pred = np.clip(model.predict(test["user"], test["item"]), 0.5, 5.0)
    return float(np.sqrt(np.mean((pred - test["rating"].to_numpy()) ** 2)))


def ranking_metrics(model, train, test, k=10, like_threshold=4.0):
    """Precision, recall and NDCG at k over items the user has not rated in the training data."""
    seen = train.groupby("user")["item"].apply(set).to_dict()
    liked = test[test["rating"] >= like_threshold].groupby("user")["item"].apply(set).to_dict()
    precisions, recalls, ndcgs = [], [], []
    for user, relevant in liked.items():
        scores = model.score_all(user).copy()
        scores[list(seen.get(user, ()))] = -np.inf
        top = np.argsort(-scores)[:k]
        hits = np.array([item in relevant for item in top], dtype=float)
        precisions.append(hits.mean())
        recalls.append(hits.sum() / len(relevant))
        dcg = (hits / np.log2(np.arange(2, k + 2))).sum()
        ideal = (1 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum()
        ndcgs.append(dcg / ideal)
    return {"precision@k": float(np.mean(precisions)), "recall@k": float(np.mean(recalls)), "ndcg@k": float(np.mean(ndcgs)), "users": len(liked)}


def to_matrix(ratings, n_users, n_items):
    return sparse.csr_matrix((ratings["rating"], (ratings["user"], ratings["item"])), shape=(n_users, n_items))
