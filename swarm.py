"""Random-mutation ensemble ('swarm'): 15 LightGBM mutants, equal-rank averaged, no selection.

Configs are drawn once (master seed 2026) and reused for every fold and the final fit. Other
params are pipeline.DEFAULT_PARAMS (lr 0.01, max_depth 6, bagging 0.8, lambda_l2 1).
Walk-forward folds 1 and 2 (refit per fold, 20-exped gap) vs lgbm-blend's recipe; metric mean
per-exped Spearman on target_everest. Only if swarm beats lgbm-blend on both folds: refit on
all train, save models/swarm, practice board once.
"""
import hashlib
import json
import sys
from pathlib import Path

import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from everestapi import EverestAPI

from build_blend import make_blend, save_components
from pipeline import DEFAULT_PARAMS, ERA, log, make_predict, to_matrix

T, GAP, N = "target_everest", 20, 15
AUX = ["target_Azegza", "target_Tamalout", "target_Aday"]
FOLDS = {"fold1": ("exped_5786", "exped_6285"), "fold2": ("exped_6286", "exped_6785")}
BLEND = {"target_everest": 17, "target_Azegza": 47, "target_Tamalout": 79, "target_Aday": 21}

rng = np.random.default_rng(2026)
MUTANTS = [dict(seed=int(rng.integers(1, 1_000_000)), feature_fraction=round(float(rng.uniform(0.3, 0.7)), 3),
                num_leaves=int(rng.integers(7, 32)), min_data_in_leaf=int(rng.integers(1000, 4001)),
                rounds=int(rng.integers(10, 41)),
                target=T if rng.random() < 0.6 else str(rng.choice(AUX))) for _ in range(N)]

c = EverestAPI()
feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
df = pd.read_parquet("data/event/pipeline_train.parquet")
feats = [f for f in feats if f in df.columns]
order = np.array(sorted(df[ERA].unique()))


def fit(a, xa, target, rounds, extra=None):
    keep = a[target].notna().to_numpy()
    y = a[target].to_numpy()[keep]
    center = 0.5 if np.nanmin(y) >= 0 and np.nanmax(y) <= 1 else float(np.mean(y))
    params = dict(DEFAULT_PARAMS, **(extra or {}))
    return lgb.train(params, lgb.Dataset(xa[keep], y - center), num_boost_round=rounds)


def mparams(m):
    return dict(seed=m["seed"], feature_fraction=m["feature_fraction"], num_leaves=m["num_leaves"],
                min_data_in_leaf=m["min_data_in_leaf"])


def score(h, p):
    s = pd.DataFrame({"e": h[ERA].to_numpy(), "p": p, "y": h[T].to_numpy()})
    return s.groupby("e").apply(lambda g: g.p.corr(g.y, method="spearman")).mean()


def avg_rank(preds):
    return pd.concat([pd.Series(p).rank(pct=True) for p in preds], axis=1).mean(axis=1).to_numpy()


print(pd.DataFrame(MUTANTS).to_string())
res, singles = {}, {}
for fold, (lo, hi) in FOLDS.items():
    h = df[(df[ERA] >= lo) & (df[ERA] <= hi)]
    a = df[df[ERA].isin(set(order[order < lo][:-GAP]))]
    xa, xh = to_matrix(a, feats, -1), to_matrix(h, feats, -1)
    res[(fold, "lgbm-blend")] = score(h, avg_rank([fit(a, xa, t, n).predict(xh) for t, n in BLEND.items()]))
    mp = [fit(a, xa, m["target"], m["rounds"], mparams(m)).predict(xh) for m in MUTANTS]
    res[(fold, "swarm")] = score(h, avg_rank(mp))
    singles[fold] = [score(h, p) for p in mp]
    log(f"{fold}: lgbm-blend {res[(fold, 'lgbm-blend')]:.4f} swarm {res[(fold, 'swarm')]:.4f}")

tab = pd.Series(res).unstack(0)[list(FOLDS)]
for fold in FOLDS:
    s = np.array(singles[fold])
    tab.loc["best single mutant (ref)", fold] = s.max()
    tab.loc["worst single mutant (ref)", fold] = s.min()
    tab.loc["median single mutant (ref)", fold] = np.median(s)
print(tab.round(4).to_string())
print("best/worst mutant index per fold:", {f: (int(np.argmax(v)), int(np.argmin(v))) for f, v in singles.items()})
tab.to_csv("data/event/cv_swarm.csv")

wins = all(res[(f, "swarm")] > res[(f, "lgbm-blend")] for f in FOLDS)
log(f"swarm beats lgbm-blend on both folds: {wins}")
if not wins:
    sys.exit(0)

xall = to_matrix(df, feats, -1)
members = [make_predict(fit(df, xall, m["target"], m["rounds"], mparams(m)).model_to_string(), feats, -1) for m in MUTANTS]
out = Path("models/swarm")
out.mkdir(parents=True, exist_ok=True)
pkl = out / "model.pkl"
pkl.write_bytes(cloudpickle.dumps(make_blend(members)))
save_components(out, [f"m{i:02d}_{m['target']}" for i, m in enumerate(MUTANTS)], members)
v = pd.read_parquet("data/event/validation.parquet")
pred = cloudpickle.loads(pkl.read_bytes())(v)
assert pred.index.equals(v.index) and pred.iloc[:, 0].between(0, 1).all()
(out / "model_meta.json").write_text(json.dumps(dict(mutants=MUTANTS, python_version="3.12",
                                                     sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))
path = Path("data/event/swarm_validation.parquet")
pd.DataFrame({"id": pred.index.astype(str), "prediction": pred.iloc[:, 0].to_numpy()}).to_parquet(path, index=False)
m = c.create_model(name="swarm")
(out / "model_id.json").write_text(json.dumps({"swarm": m["id"]}))
r = c.submit_validation_diagnostics(m["id"], str(path), target=T, model_pkl=str(pkl), model_pkl_python_version="3.12",
                                    client_label="swarm practice", wait=True, timeout=600)
print("swarm", json.dumps({k: r.get(k) for k in ["upload_id", "status", "error", "fit20", "unq", "inov", "round_score"]},
                          default=str))
