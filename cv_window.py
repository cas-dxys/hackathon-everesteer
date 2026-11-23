"""Training-window length check: the shipped 17-round DEFAULT_PARAMS model on target_everest.

For each test block (fold 1: expeds 5786-6285, fold 2: 6286-6785) fit on the W expeds ending
20 before it, W in {1000, 2000, 3500, all}, plus 'all' with sample weights doubling every
1000 expeds toward the present. Metric: mean per-exped Spearman.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from everestapi import EverestAPI
from pipeline import DEFAULT_PARAMS, ERA, log, to_matrix

T, GAP, ROUNDS = "target_everest", 20, 17
FOLDS = {"fold1": ("exped_5786", "exped_6285"), "fold2": ("exped_6286", "exped_6785")}
OPTIONS = ["w1000", "w2000", "w3500", "all", "all_recency"]

feats = list(EverestAPI().get_dataset_schema(verbose=True)["feature_sets"]["all"])
df = pd.read_parquet("data/event/pipeline_train.parquet")
df = df[df[T].notna()]
feats = [f for f in feats if f in df.columns]
order = np.array(sorted(df[ERA].unique()))
pos = {e: i for i, e in enumerate(order)}


def recency_weight(expeds: pd.Series) -> np.ndarray:
    idx = expeds.map(pos).to_numpy()
    return 2.0 ** ((idx - idx.max()) / 1000.0)


def score(h, p):
    s = pd.DataFrame({"e": h[ERA].to_numpy(), "p": p, "y": h[T].to_numpy()})
    return s.groupby("e").apply(lambda g: g.p.corr(g.y, method="spearman")).mean()


res = {}
for fold, (lo, hi) in FOLDS.items():
    h = df[(df[ERA] >= lo) & (df[ERA] <= hi)]
    xh = to_matrix(h, feats, -1)
    before = order[order < lo][:-GAP]
    for opt in OPTIONS:
        fit_e = before[-int(opt[1:]):] if opt.startswith("w") else before
        a = df[df[ERA].isin(set(fit_e))]
        w = recency_weight(a[ERA]) if opt == "all_recency" else None
        b = lgb.train(DEFAULT_PARAMS, lgb.Dataset(to_matrix(a, feats, -1), a[T].to_numpy() - 0.5, weight=w),
                      num_boost_round=ROUNDS)
        res[(fold, opt)] = score(h, b.predict(xh))
        log(f"{fold} {opt}: {len(fit_e)} expeds -> {res[(fold, opt)]:.4f}")

tab = pd.Series(res).unstack(0)[list(FOLDS)].loc[OPTIONS]
tab["fold2_vs_all"] = tab.fold2 - tab.loc["all", "fold2"]
tab["fold1_vs_all"] = tab.fold1 - tab.loc["all", "fold1"]
print(tab.round(4).to_string())
tab.to_csv("data/event/cv_window.csv")
