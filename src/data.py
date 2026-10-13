import numpy as np
import pandas as pd


def generate_ratings(n_users=1500, n_items=400, density=0.04, n_factors=6, seed=81):
    """Ratings from a low-rank taste model plus user/item biases; popular items get more ratings."""
    rng = np.random.default_rng(seed)
    user_f = rng.normal(0, 0.8, (n_users, n_factors))
    item_f = rng.normal(0, 0.8, (n_items, n_factors))
    user_bias = rng.normal(0, 0.35, n_users)
    item_bias = rng.normal(0, 0.45, n_items)

    popularity = rng.pareto(1.2, n_items) + 1
    popularity = popularity / popularity.sum()
    activity = rng.pareto(1.5, n_users) + 1
    activity = activity / activity.sum()
    n_ratings = int(n_users * n_items * density)

    users = rng.choice(n_users, n_ratings, p=activity)
    items = rng.choice(n_items, n_ratings, p=popularity)
    pairs = pd.DataFrame({"user": users, "item": items}).drop_duplicates()

    raw = (
        3.5
        + user_bias[pairs["user"]]
        + item_bias[pairs["item"]]
        + np.einsum("ij,ij->i", user_f[pairs["user"]], item_f[pairs["item"]])
        + rng.normal(0, 0.4, len(pairs))
    )
    pairs["rating"] = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)
    return pairs.reset_index(drop=True)
