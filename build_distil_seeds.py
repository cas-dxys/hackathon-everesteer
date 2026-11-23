"""distil-seeds: equal-rank average of 5 LightGBMs with distil's exact config and target (the event
benchmark's train predictions, aligned by id, centred at their mean), seeds 1-5; only `seed`
changes. Prints pairwise correlation on round_2 live rows, then practice board once.
"""
import hashlib
import itertools
import json
from pathlib import Path

import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from everestapi import EverestAPI

from build_blend import make_blend, save_components
from pipeline import log, make_predict, to_matrix

DATA = Path("data/event")
meta = json.loads(Path("models/distil/model_meta.json").read_text())
PARAMS, ROUNDS = meta["params"], meta["num_boost_round"]
COL = meta["target"].split(":", 1)[1]
SEEDS = [1, 2, 3, 4, 5]

c = EverestAPI()
feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
train = pd.read_parquet(DATA / "pipeline_train.parquet")
feats = [f for f in feats if f in train.columns]
bench = pd.read_parquet(DATA / "bench" / "bench_train.parquet")
y = bench[COL].reindex(train.index)
keep = y.notna().to_numpy()
x = to_matrix(train[keep], feats, -1)
yc = y[keep].to_numpy() - float(y[keep].mean())
log(f"target {COL}: {keep.sum()}/{len(train)} rows, {ROUNDS} rounds")
members = []
for s in SEEDS:
    b = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(x, yc), num_boost_round=ROUNDS)
    members.append(make_predict(b.model_to_string(), feats, -1))
    log(f"seed {s} done")

live = pd.read_parquet(DATA / "round_round_2_live.parquet")
p = pd.concat([m(live).iloc[:, 0].rename(f"s{s}") for s, m in zip(SEEDS, members)], axis=1)
pairs = list(itertools.combinations(p.columns, 2))
overall = [p[a].corr(p[b], method="spearman") for a, b in pairs]
log(f"round_2 live: mean pairwise spearman {np.mean(overall):.3f} (min {min(overall):.3f}, max {max(overall):.3f})")
dp = cloudpickle.loads(Path("models/distil/model.pkl").read_bytes())(live).iloc[:, 0]
log("each seed vs saved distil (seed 7): " + ", ".join(f"{col} {p[col].corr(dp, method='spearman'):.3f}" for col in p))

target = c.get_dataset_schema()["primary_target"]
out = Path("models/distil-seeds")
out.mkdir(parents=True, exist_ok=True)
pkl = out / "model.pkl"
pkl.write_bytes(cloudpickle.dumps(make_blend(members)))
save_components(out, [f"seed{s}" for s in SEEDS], members)
blend = cloudpickle.loads(pkl.read_bytes())
v = pd.read_parquet(DATA / "validation.parquet")
pred = blend(v)
assert pred.index.equals(v.index) and pred.iloc[:, 0].between(0, 1).all()
lv = blend(live)
assert lv.index.equals(live.index) and lv.iloc[:, 0].between(0, 1).all()
(out / "model_meta.json").write_text(json.dumps(dict(members={f"seed{s}": ROUNDS for s in SEEDS}, seeds=SEEDS,
                                                     target=meta["target"], params=PARAMS, num_boost_round=ROUNDS,
                                                     python_version="3.12",
                                                     sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))
path = DATA / "distil_seeds_validation.parquet"
pd.DataFrame({"id": pred.index.astype(str), "prediction": pred.iloc[:, 0].to_numpy()}).to_parquet(path, index=False)
m = c.create_model(name="distil-seeds")
(out / "model_id.json").write_text(json.dumps({"distil-seeds": m["id"]}))
r = c.submit_validation_diagnostics(m["id"], str(path), target=target, model_pkl=str(pkl), model_pkl_python_version="3.12",
                                    client_label="distil-seeds practice", wait=True, timeout=600)
print("distil-seeds", json.dumps({k: r.get(k) for k in ["upload_id", "status", "error", "fit20", "unq", "inov",
                                                        "round_score"]}, default=str))
print("public name:", next(x["name"] for x in c.get_models()["models"] if x["id"] == m["id"]))
