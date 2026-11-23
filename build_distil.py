"""distil: lgbm-main's config (DEFAULT_PARAMS, 17 rounds) fit on the event benchmark's train
predictions (download_benchmark("futures", "train"), aligned by id) instead of target_everest.
distil-aux: equal-rank average of distil and aux3's 3 components. Practice board once each.
"""
import hashlib
import json
from pathlib import Path

import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from everestapi import EverestAPI

from build_blend import make_blend, save_components
from pipeline import DEFAULT_PARAMS, log, make_predict, to_matrix

ROUNDS = json.loads(Path("models/lgbm-main/model_meta.json").read_text())["num_boost_round"]
AUX = ["target_Azegza", "target_Tamalout", "target_Aday"]
DATA = Path("data/event")
BENCH = DATA / "bench"

c = EverestAPI()
target = c.get_dataset_schema()["primary_target"]
feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
train = pd.read_parquet(DATA / "pipeline_train.parquet")
feats = [f for f in feats if f in train.columns]
bench = pd.read_parquet(c.download_benchmark("futures", "train", output_path=str(BENCH / "bench_train.parquet")))
col = [x for x in bench.columns if x != "exped"][0]
y = bench[col].reindex(train.index)
keep = y.notna().to_numpy()
log(f"benchmark column {col}: {keep.sum()}/{len(train)} rows aligned, range [{y.min():.3f}, {y.max():.3f}]")
b = lgb.train(DEFAULT_PARAMS, lgb.Dataset(to_matrix(train[keep], feats, -1), y[keep].to_numpy() - float(y[keep].mean())),
              num_boost_round=ROUNDS)
distil = make_predict(b.model_to_string(), feats, -1)

v = pd.read_parquet(DATA / "validation.parquet")
members = [distil] + [cloudpickle.loads(Path(f"models/aux3/components/{t}.pkl").read_bytes()) for t in AUX]
for name, predict, meta in [
    ("distil", distil, dict(target=f"benchmark:{col}", params=DEFAULT_PARAMS, num_boost_round=ROUNDS)),
    ("distil-aux", make_blend(members), dict(members=["distil", *AUX])),
]:
    out = Path("models") / name
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "model.pkl"
    pkl.write_bytes(cloudpickle.dumps(predict))
    if name == "distil-aux":
        save_components(out, ["distil", *AUX], members)
    pred = cloudpickle.loads(pkl.read_bytes())(v)
    assert pred.index.equals(v.index) and pred.iloc[:, 0].between(0, 1).all()
    (out / "model_meta.json").write_text(json.dumps(dict(meta, python_version="3.12",
                                                         sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))
    path = DATA / f"{name.replace('-', '_')}_validation.parquet"
    pd.DataFrame({"id": pred.index.astype(str), "prediction": pred.iloc[:, 0].to_numpy()}).to_parquet(path, index=False)
    m = c.create_model(name=name)
    (out / "model_id.json").write_text(json.dumps({name: m["id"]}))
    r = c.submit_validation_diagnostics(m["id"], str(path), target=target, model_pkl=str(pkl),
                                        model_pkl_python_version="3.12", client_label=f"{name} practice",
                                        wait=True, timeout=600)
    print(name, json.dumps({k: r.get(k) for k in ["upload_id", "status", "error", "fit20", "unq", "inov",
                                                  "round_score"]}, default=str), flush=True)
