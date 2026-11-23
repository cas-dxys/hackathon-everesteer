"""lgbm-base2: the hackathon starter.py recipe on event train, in our predict() convention,
plus blend2 = equal-rank average of lgbm-base2 and lgbm-blend's 3 aux components.

starter.py recipe, verbatim: features = sorted(train columns starting 'feature_'), the schema's
missing sentinel masked to NaN, rows with a NaN target dropped, raw target (not centred),
LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).
Ours wraps it as a closure over the booster text with a rank(pct=True) output.
"""
import hashlib
import json
from pathlib import Path

import cloudpickle
import lightgbm as lgb
import pandas as pd
from everestapi import EverestAPI

from build_blend import make_blend, save_components
from pipeline import log, make_predict

AUX = ["target_Azegza", "target_Tamalout", "target_Aday"]
DATA = Path("data/event")

c = EverestAPI()
schema = c.get_dataset_schema()
target = schema["primary_target"]
missing = (schema.get("feature_encoding") or {}).get("missing")
missing = -1.0 if missing is None else missing
train = pd.read_parquet(DATA / "pipeline_train.parquet")
feat_cols = sorted(col for col in train.columns if col.startswith("feature_"))
fit = train.dropna(subset=[target])
log(f"starter recipe: {len(fit)} rows, {len(feat_cols)} features, missing={missing}")
x = fit[feat_cols].astype("float32")
model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
model.fit(x.mask(x == missing), fit[target])
base2 = make_predict(model.booster_.model_to_string(), feat_cols, missing)

members = [base2] + [cloudpickle.loads((Path("models/lgbm-blend/components") / f"{t}.pkl").read_bytes()) for t in AUX]
v = pd.read_parquet(DATA / "validation.parquet")
for name, predict, meta in [
    ("lgbm-base2", base2, dict(recipe="hackathon starter.py", n_estimators=400, learning_rate=0.05,
                               num_leaves=31, n_features=len(feat_cols), missing=missing, target=target)),
    ("blend2", make_blend(members), dict(members=["lgbm-base2", *AUX])),
]:
    out = Path("models") / name
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "model.pkl"
    pkl.write_bytes(cloudpickle.dumps(predict))
    if name == "blend2":
        save_components(out, ["lgbm-base2", *AUX], members)
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
