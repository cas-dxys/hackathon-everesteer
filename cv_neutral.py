"""Per-exped feature neutralisation of lgbm-blend, walk-forward folds 1 and 2 (local only).

Components refit per fold on expeds ending 20 before the test block, at lgbm-blend's round
counts. Neutralise: per exped, regress the blend's centred rank on features (centred bins,
-1 -> 0, plus intercept), subtract p x fitted part, re-rank. Two feature sets: all 178 (as
specified; with ~91 rows per exped this is rank-deficient, so the fit is exact) and the top
20 by lgbm-main gain. Metric: mean per-exped Spearman on target_everest.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from everestapi import EverestAPI

from pipeline import DEFAULT_PARAMS, ERA, log, to_matrix

T, GAP = "target_everest", 20
FOLDS = {"fold1": ("exped_5786", "exped_6285"), "fold2": ("exped_6286", "exped_6785")}
MEMBERS = json.load(open("models/lgbm-blend/model_meta.json"))["members"]  # name -> rounds
MEMBERS = {("target_everest" if k == "lgbm-main" else k): v for k, v in MEMBERS.items()}
PS = [0.0, 0.25, 0.5, 0.75]

feats = list(EverestAPI().get_dataset_schema(verbose=True)["feature_sets"]["all"])
df = pd.read_parquet("data/event/pipeline_train.parquet")
feats = [f for f in feats if f in df.columns]
order = np.array(sorted(df[ERA].unique()))

main_b = lgb.Booster(model_file="models/lgbm-blend/components/lgbm-main.txt")
gain = pd.Series(main_b.feature_importance("gain"), index=feats).sort_values(ascending=False)
share = gain / gain.sum()
print(f"feature_Ouzoud in lgbm-main: gain rank {list(gain.index).index('feature_Ouzoud') + 1}/{len(feats)}, "
      f"gain share {share['feature_Ouzoud']:.4f}, splits {int(pd.Series(main_b.feature_importance('split'), index=feats)['feature_Ouzoud'])}")
print("top 5 by gain:", share.head(5).round(3).to_dict())
TOP20 = list(gain.index[:20])


def centred(d, cols):
    x = d[cols].to_numpy(dtype=np.float64)
    z = (x - 4.5) / 9.0
    z[x == -1] = 0.0
    return z


def neutralise(h, pred, cols, p):
    out = pd.Series(index=h.index, dtype=float)
    z_all = centred(h, cols)
    for e, idx in h.groupby(ERA).indices.items():
        y = pd.Series(pred[idx]).rank(pct=True).to_numpy() - 0.5
        X = np.column_stack([np.ones(len(idx)), z_all[idx]])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.iloc[idx] = pd.Series(y - p * (X @ beta - beta[0])).rank(pct=True).to_numpy()
    return out.to_numpy()


def score(h, p):
    s = pd.DataFrame({"e": h[ERA].to_numpy(), "p": p, "y": h[T].to_numpy()})
    return s.groupby("e").apply(lambda g: g.p.corr(g.y, method="spearman")).mean()


res = {}
for fold, (lo, hi) in FOLDS.items():
    h = df[(df[ERA] >= lo) & (df[ERA] <= hi)]
    xh = to_matrix(h, feats, -1)
    fit_e = set(order[order < lo][:-GAP])
    preds = []
    for t, n in MEMBERS.items():
        a = df[df[ERA].isin(fit_e) & df[t].notna()]
        center = 0.5 if a[t].between(0, 1).all() else float(a[t].mean())
        b = lgb.train(DEFAULT_PARAMS, lgb.Dataset(to_matrix(a, feats, -1), a[t].to_numpy() - center), num_boost_round=n)
        preds.append(pd.Series(b.predict(xh)).rank(pct=True))
    blend = pd.concat(preds, axis=1).mean(axis=1).to_numpy()
    log(f"{fold}: rows/exped {len(h) / h[ERA].nunique():.0f}, lgbm-main alone {score(h, preds[0].to_numpy()):.4f}")
    for name, cols in [("all178", feats), ("top20", TOP20)]:
        for p in PS:
            res[(fold, f"{name}_p{p}")] = score(h, neutralise(h, blend, cols, p))
            log(f"{fold} {name} p={p}: {res[(fold, f'{name}_p{p}')]:.4f}")

tab = pd.Series(res).unstack(0)[list(FOLDS)]
print(tab.round(4).to_string())
tab.to_csv("data/event/cv_neutral.csv")
