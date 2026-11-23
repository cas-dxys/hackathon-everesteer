"""aux3: equal-rank average of lgbm-blend's 3 aux components only (no lgbm-main).

Folds 1 and 2 exactly as cv_neutral.py / swarm.py (refit per fold, 20-exped gap, lgbm-blend's
round counts). Gate: aux3 > lgbm-blend on fold 2 and aux3 >= 0.03 on fold 1. If it passes,
models/aux3 is built from the SAME full-train components as lgbm-blend (no refit) and sent to
the practice board once.
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
from pipeline import DEFAULT_PARAMS, ERA, log, to_matrix

T, GAP = "target_everest", 20
FOLDS = {"fold1": ("exped_5786", "exped_6285"), "fold2": ("exped_6286", "exped_6785")}
BLEND = {"target_everest": 17, "target_Azegza": 47, "target_Tamalout": 79, "target_Aday": 21}
AUX = ["target_Azegza", "target_Tamalout", "target_Aday"]

c = EverestAPI()
feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
df = pd.read_parquet("data/event/pipeline_train.parquet")
feats = [f for f in feats if f in df.columns]
order = np.array(sorted(df[ERA].unique()))


def score(h, p):
    s = pd.DataFrame({"e": h[ERA].to_numpy(), "p": p, "y": h[T].to_numpy()})
    return s.groupby("e").apply(lambda g: g.p.corr(g.y, method="spearman")).mean()


def avg_rank(preds):
    return pd.concat([pd.Series(p).rank(pct=True) for p in preds], axis=1).mean(axis=1).to_numpy()


res = {}
for fold, (lo, hi) in FOLDS.items():
    h = df[(df[ERA] >= lo) & (df[ERA] <= hi)]
    a = df[df[ERA].isin(set(order[order < lo][:-GAP]))]
    xa, xh = to_matrix(a, feats, -1), to_matrix(h, feats, -1)
    p = {}
    for t, n in BLEND.items():
        keep = a[t].notna().to_numpy()
        b = lgb.train(DEFAULT_PARAMS, lgb.Dataset(xa[keep], a[t].to_numpy()[keep] - 0.5), num_boost_round=n)
        p[t] = b.predict(xh)
    res[(fold, "lgbm-blend")] = score(h, avg_rank(list(p.values())))
    res[(fold, "aux3")] = score(h, avg_rank([p[t] for t in AUX]))
    for t in AUX:
        res[(fold, f"  {t} alone (ref)")] = score(h, p[t])
    log(f"{fold}: lgbm-blend {res[(fold, 'lgbm-blend')]:.4f} aux3 {res[(fold, 'aux3')]:.4f}")

tab = pd.Series(res).unstack(0)[list(FOLDS)]
print(tab.round(4).to_string())
tab.to_csv("data/event/cv_aux3.csv")
ok = res[("fold2", "aux3")] > res[("fold2", "lgbm-blend")] and res[("fold1", "aux3")] >= 0.03
log(f"gate (fold2 > lgbm-blend and fold1 >= 0.03): {ok}")
if not ok:
    sys.exit(0)

members = [cloudpickle.loads(Path(f"models/lgbm-blend/components/{t}.pkl").read_bytes()) for t in AUX]
out = Path("models/aux3")
out.mkdir(parents=True, exist_ok=True)
pkl = out / "model.pkl"
pkl.write_bytes(cloudpickle.dumps(make_blend(members)))
save_components(out, AUX, members)
v = pd.read_parquet("data/event/validation.parquet")
pred = cloudpickle.loads(pkl.read_bytes())(v)
assert pred.index.equals(v.index) and pred.iloc[:, 0].between(0, 1).all()
(out / "model_meta.json").write_text(json.dumps(dict(members={t: BLEND[t] for t in AUX}, python_version="3.12",
                                                     sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))
path = Path("data/event/aux3_validation.parquet")
pd.DataFrame({"id": pred.index.astype(str), "prediction": pred.iloc[:, 0].to_numpy()}).to_parquet(path, index=False)
m = c.create_model(name="aux3")
(out / "model_id.json").write_text(json.dumps({"aux3": m["id"]}))
r = c.submit_validation_diagnostics(m["id"], str(path), target=T, model_pkl=str(pkl), model_pkl_python_version="3.12",
                                    client_label="aux3 practice", wait=True, timeout=600)
print("aux3", json.dumps({k: r.get(k) for k in ["upload_id", "status", "error", "fit20", "unq", "inov", "round_score"]},
                         default=str))
