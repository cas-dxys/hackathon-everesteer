"""main-swarm: equal-rank average of 20 LightGBMs on target_everest, all on the full event train at
lgbm-main's learning rate and round count, each with a mild jitter of lgbm-main's config:
seed i (1-20), feature_fraction U(0.4, 0.6), min_data_in_leaf U{1500..3000}, num_leaves U{23..31}.
Draws come from a fixed master seed. No selection, no dropping. Practice board is not used.
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
rng = np.random.default_rng(2027)
MEMBERS = [dict(seed=i, feature_fraction=round(float(rng.uniform(0.4, 0.6)), 3),
                min_data_in_leaf=int(rng.integers(1500, 3001)), num_leaves=int(rng.integers(23, 32)))
           for i in range(1, 21)]

c = EverestAPI()
feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
train = pd.read_parquet(DATA / "pipeline_train.parquet")
train = train[train[TARGET].notna()]
feats = [f for f in feats if f in train.columns]
x = to_matrix(train, feats, MISSING)
y = train[TARGET].to_numpy() - 0.5
members = []
for mp in MEMBERS:
    b = lgb.train(dict(DEFAULT_PARAMS, **mp), lgb.Dataset(x, y), num_boost_round=ROUNDS)
    members.append(make_predict(b.model_to_string(), feats, MISSING))
log(f"trained {len(members)} members, {ROUNDS} rounds, lr {DEFAULT_PARAMS['learning_rate']}")
print(pd.DataFrame(MEMBERS).to_string(index=False))

out = Path("models/main-swarm")
out.mkdir(parents=True, exist_ok=True)
pkl = out / "model.pkl"
pkl.write_bytes(cloudpickle.dumps(make_blend(members)))
save_components(out, [f"m{mp['seed']:02d}" for mp in MEMBERS], members)
(out / "model_meta.json").write_text(json.dumps(dict(members=MEMBERS, base_params=DEFAULT_PARAMS, num_boost_round=ROUNDS,
                                                     target=TARGET, python_version="3.12",
                                                     sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))
blend = cloudpickle.loads(pkl.read_bytes())
live = pd.read_parquet(DATA / "round_round_3_live.parquet")
sw = blend(live)
assert sw.index.equals(live.index) and sw.iloc[:, 0].between(0, 1).all() and np.isfinite(sw.iloc[:, 0]).all()
sw_path = DATA / "round_round_3_live_main-swarm.parquet"
pd.DataFrame({"id": sw.index.astype(str), "prediction": sw.iloc[:, 0].to_numpy()}).to_parquet(sw_path, index=False)

ms = cloudpickle.loads(Path("models/main-seeds/model.pkl").read_bytes())(live).iloc[:, 0]
mem = pd.concat([m(live).iloc[:, 0].rename(i) for i, m in enumerate(members)], axis=1)
pairs = [mem[a].corr(mem[b], method="spearman") for a, b in itertools.combinations(mem.columns, 2)]
vs_seeds = [mem[col].corr(ms, method="spearman") for col in mem]
per = pd.DataFrame({"e": live["exped"], "a": sw.iloc[:, 0], "b": ms}).groupby("e").apply(
    lambda g: g.a.corr(g.b, method="spearman"))
log(f"round_3 live {len(live)} rows: main-swarm vs main-seeds spearman {sw.iloc[:, 0].corr(ms, method='spearman'):.3f} "
    f"(per-exped mean {per.mean():.3f}) | swarm members pairwise mean {np.mean(pairs):.3f} "
    f"(min {min(pairs):.3f}) | each member vs main-seeds mean {np.mean(vs_seeds):.3f}")
m = c.create_model(name="main-swarm")
(out / "model_id.json").write_text(json.dumps({"main-swarm": m["id"]}))
log(f"model main-swarm id={m['id']} public name="
    f"{next(x['name'] for x in c.get_models()['models'] if x['id'] == m['id'])}; predictions -> {sw_path}")
