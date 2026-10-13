import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    ratings = data.generate_ratings()
    n_users, n_items = ratings["user"].max() + 1, ratings["item"].max() + 1
    train, test = core.train_test_split_ratings(ratings)

    models = {
        "popularity": core.Popularity().fit(train, n_users, n_items),
        "bias_model": core.BiasModel().fit(train, n_users, n_items),
        "als": core.ALS(factors=8, reg=6.0, iters=12).fit(train, n_users, n_items),
    }
    rows = {}
    for name, model in models.items():
        rows[name] = {"rmse": core.rmse(model, test), **core.ranking_metrics(model, train, test)}
    table = pd.DataFrame(rows).T

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    table["rmse"].plot.bar(ax=axes[0], color="grey", rot=0)
    axes[0].set_title("RMSE on held-out ratings (lower is better)")
    table[["precision@k", "recall@k", "ndcg@k"]].plot.bar(ax=axes[1], rot=0, color=["lightgrey", "grey", "black"])
    axes[1].set_title("Top-10 ranking quality")
    fig.tight_layout()
    fig.savefig(OUT / "recommender.png", dpi=130)

    table.round(4).to_csv(OUT / "metrics.csv")
    sparsity = 1 - len(ratings) / (n_users * n_items)
    summary = {
        "users": int(n_users),
        "items": int(n_items),
        "ratings": int(len(ratings)),
        "sparsity": round(float(sparsity), 3),
        "test_ratings": int(len(test)),
        "metrics": {k: {m: round(float(v), 4) for m, v in row.items()} for k, row in table.iterrows()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
