"""main-distil: equal-rank average of the saved lgbm-main and distil pkls (no retraining).
Prints their correlation on round_1 live rows, then practice board once.
"""
import hashlib
import json
from pathlib import Path

import cloudpickle
import pandas as pd
from everestapi import EverestAPI

from build_blend import make_blend, save_components
from pipeline import log

DATA = Path("data/event")
NAMES = ["lgbm-main", "distil"]
members = [cloudpickle.loads((Path("models") / n / "model.pkl").read_bytes()) for n in NAMES]

live = pd.read_parquet(DATA / "round_round_1_live.parquet")
p = pd.concat([m(live).iloc[:, 0].rename(n) for n, m in zip(NAMES, members)], axis=1)
p["exped"] = live["exped"]
per = p.groupby("exped").apply(lambda g: g["lgbm-main"].corr(g["distil"], method="spearman"))
log(f"round_1 live: {len(live)} rows, {live['exped'].nunique()} expeds | lgbm-main vs distil spearman: "
    f"overall {p['lgbm-main'].corr(p['distil'], method='spearman'):.3f}, per-exped mean {per.mean():.3f} "
    f"(min {per.min():.3f}, max {per.max():.3f}) | pearson {p['lgbm-main'].corr(p['distil']):.3f}")

c = EverestAPI()
target = c.get_dataset_schema()["primary_target"]
out = Path("models/main-distil")
out.mkdir(parents=True, exist_ok=True)
pkl = out / "model.pkl"
pkl.write_bytes(cloudpickle.dumps(make_blend(members)))
save_components(out, NAMES, members)
v = pd.read_parquet(DATA / "validation.parquet")
pred = cloudpickle.loads(pkl.read_bytes())(v)
assert pred.index.equals(v.index) and pred.iloc[:, 0].between(0, 1).all()
lv = cloudpickle.loads(pkl.read_bytes())(live)
assert lv.index.equals(live.index) and lv.iloc[:, 0].between(0, 1).all()
(out / "model_meta.json").write_text(json.dumps(dict(members=NAMES, python_version="3.12",
                                                     sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))
path = DATA / "main_distil_validation.parquet"
pd.DataFrame({"id": pred.index.astype(str), "prediction": pred.iloc[:, 0].to_numpy()}).to_parquet(path, index=False)
m = c.create_model(name="main-distil")
(out / "model_id.json").write_text(json.dumps({"main-distil": m["id"]}))
r = c.submit_validation_diagnostics(m["id"], str(path), target=target, model_pkl=str(pkl), model_pkl_python_version="3.12",
                                    client_label="main-distil practice", wait=True, timeout=600)
print("main-distil", json.dumps({k: r.get(k) for k in ["upload_id", "status", "error", "fit20", "unq", "inov",
                                                       "round_score"]}, default=str))
print("public name:", next(x["name"] for x in c.get_models()["models"] if x["id"] == m["id"]))
