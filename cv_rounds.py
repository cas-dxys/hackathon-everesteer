"""Walk-forward check of fixed boosting rounds on the event train split (target_everest only).

3 folds over the last ~1,500 expeds, each fold's test block 500 expeds, fitted on every
exped before (test start - 20). Metric: mean per-exped Spearman of prediction vs target.
Compares the shipped 17-round DEFAULT_PARAMS model with lr 0.02 at {100, 300, 600} rounds.
"""
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from everestapi import EverestAPI
from pipeline import DEFAULT_PARAMS, ERA, log, to_matrix

T, GAP, N_TEST, K = "target_everest", 20, 500, 3
CANDIDATE = dict(DEFAULT_PARAMS, learning_rate=0.02, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.5)
ROUNDS = [100, 300, 600]

feats = list(EverestAPI().get_dataset_schema(verbose=True)["feature_sets"]["all"])
df = pd.read_parquet("data/event/pipeline_train.parquet")
df = df[df[T].notna()]
feats = [f for f in feats if f in df.columns]
order = np.array(sorted(df[ERA].unique()))


def score(h, p):
    s = pd.DataFrame({"e": h[ERA].to_numpy(), "p": p, "y": h[T].to_numpy()})
    return s.groupby("e").apply(lambda g: g.p.corr(g.y, method="spearman")).mean()


rows = []
for k in range(K):
    test_e = order[len(order) - (K - k) * N_TEST: len(order) - (K - k - 1) * N_TEST]
    fit_e = order[order < test_e[0]][:-GAP]
    a, h = df[df[ERA].isin(fit_e)], df[df[ERA].isin(test_e)]
    xa, xh = to_matrix(a, feats, -1), to_matrix(h, feats, -1)
    y = a[T].to_numpy() - 0.5
    base = lgb.train(DEFAULT_PARAMS, lgb.Dataset(xa, y), num_boost_round=17)
    r = {"fold": k, "test": f"{test_e[0]}..{test_e[-1]}", "fit_expeds": len(fit_e),
         "baseline_r17": score(h, base.predict(xh))}
    cand = lgb.train(CANDIDATE, lgb.Dataset(xa, y), num_boost_round=max(ROUNDS))
    for n in ROUNDS:
        r[f"lr02_r{n}"] = score(h, cand.predict(xh, num_iteration=n))
    log(json.dumps(r, default=float))
    rows.append(r)

res = pd.DataFrame(rows).set_index("fold")
print(res.round(4).to_string())
num = res.drop(columns=["test", "fit_expeds"])
print("\nmean:\n" + num.mean().round(4).to_string() + "\n\nstd:\n" + num.std().round(4).to_string())
res.to_csv("data/event/cv_rounds.csv")
