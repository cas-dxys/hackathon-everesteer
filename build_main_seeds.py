"""main-seeds: equal-rank average of 5 LightGBMs with lgbm-main's exact config and training data
(pipeline.DEFAULT_PARAMS, lgbm-main's num_boost_round, all event train, target - 0.5), seeds 1-5;
only `seed` changes. Prints pairwise correlation on round_2 live rows, then practice board once.
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
from pipeline import DEFAULT_PARAMS, log, make_predict, to_matrix

DATA = Path("data/event")
main_meta = json.loads(Path("models/lgbm-main/model_meta.json").read_text())
ROUNDS, TARGET, MISSING = main_meta["num_boost_round"], main_meta["target"], main_meta["missing"]
assert main_meta["params"] == DEFAULT_PARAMS, "lgbm-main was not trained on DEFAULT_PARAMS"
SEEDS = [1, 2, 3, 4, 5]

c = EverestAPI()
feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
train = pd.read_parquet(DATA / "pipeline_train.parquet")
train = train[train[TARGET].notna()]
feats = [f for f in feats if f in train.columns]
assert len(feats) == main_meta["n_features"]
x = to_matrix(train, feats, MISSING)
center = 0.5 if train[TARGET].between(0, 1).all() else float(train[TARGET].mean())
members = []
for s in SEEDS:
    b = lgb.train(dict(DEFAULT_PARAMS, seed=s), lgb.Dataset(x, train[TARGET].to_numpy() - center), num_boost_round=ROUNDS)
    members.append(make_predict(b.model_to_string(), feats, MISSING))
    log(f"seed {s}: {ROUNDS} rounds on {len(train)} rows")

live = pd.read_parquet(DATA / "round_round_2_live.parquet")
p = pd.concat([m(live).iloc[:, 0].rename(f"s{s}") for s, m in zip(SEEDS, members)], axis=1)
pairs = list(itertools.combinations(p.columns, 2))
overall = [p[a].corr(p[b], method="spearman") for a, b in pairs]
per = [p.groupby(live["exped"]).apply(lambda g: g[a].corr(g[b], method="spearman")).mean() for a, b in pairs]
log(f"round_2 live: {len(live)} rows, {live['exped'].nunique()} expeds | mean pairwise spearman over {len(pairs)} pairs: "
    f"overall {np.mean(overall):.3f} (min {min(overall):.3f}, max {max(overall):.3f}), per-exped mean {np.mean(per):.3f}")
mainp = cloudpickle.loads(Path("models/lgbm-main/model.pkl").read_bytes())(live).iloc[:, 0]
log("each seed vs saved lgbm-main (seed 7): " + ", ".join(f"{col} {p[col].corr(mainp, method='spearman'):.3f}" for col in p))

target = c.get_dataset_schema()["primary_target"]
out = Path("models/main-seeds")
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
                                                     params=DEFAULT_PARAMS, num_boost_round=ROUNDS, python_version="3.12",
                                                     sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))
path = DATA / "main_seeds_validation.parquet"
pd.DataFrame({"id": pred.index.astype(str), "prediction": pred.iloc[:, 0].to_numpy()}).to_parquet(path, index=False)
m = c.create_model(name="main-seeds")
(out / "model_id.json").write_text(json.dumps({"main-seeds": m["id"]}))
r = c.submit_validation_diagnostics(m["id"], str(path), target=target, model_pkl=str(pkl), model_pkl_python_version="3.12",
                                    client_label="main-seeds practice", wait=True, timeout=600)
print("main-seeds", json.dumps({k: r.get(k) for k in ["upload_id", "status", "error", "fit20", "unq", "inov",
                                                      "round_score"]}, default=str))
print("public name:", next(x["name"] for x in c.get_models()["models"] if x["id"] == m["id"]))
