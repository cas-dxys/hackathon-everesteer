"""ridge-ev: eval.py's ridge (alpha 1e4, bins centred (x-4.5)/9, -1 -> 0, target - 0.5) on event
train, as a numpy closure over its coefficients with a rank(pct=True) output; plus blend3 =
equal-rank average of ridge-ev and lgbm-blend's 3 aux components. Practice board once each.
"""
import hashlib
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from everestapi import EverestAPI
from sklearn.linear_model import Ridge

from build_blend import make_blend
from eval import MISSING, design_matrix
from pipeline import log

AUX = ["target_Azegza", "target_Tamalout", "target_Aday"]
DATA = Path("data/event")


def make_ridge_predict(coef, intercept, features):
    def predict(live_features):
        import numpy as np
        import pandas as pd

        raw = live_features.reindex(columns=features).to_numpy(dtype=np.float32)
        x = (raw - 4.5) / 9.0
        x[(raw == -1) | np.isnan(raw)] = 0.0
        out = pd.Series(x @ coef + intercept, index=live_features.index)
        return out.rank(pct=True).clip(0.0, 1.0).to_frame("prediction")

    return predict


c = EverestAPI()
target = c.get_dataset_schema()["primary_target"]
feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
train = pd.read_parquet(DATA / "pipeline_train.parquet")
feats = [f for f in feats if f in train.columns]
assert MISSING == -1
y = train[target].to_numpy(dtype=np.float64) - 0.5
keep = ~np.isnan(y)
model = Ridge(alpha=1e4, fit_intercept=True).fit(design_matrix(train, feats)[keep], y[keep])
log(f"ridge fit: {keep.sum()} rows, {len(feats)} features")
ridge = make_ridge_predict(model.coef_.astype(np.float64), float(model.intercept_), feats)

v = pd.read_parquet(DATA / "validation.parquet")
check = pd.Series(model.predict(design_matrix(v, feats)), index=v.index).rank(pct=True)
assert np.allclose(check.to_numpy(), ridge(v).iloc[:, 0].to_numpy(), atol=1e-4), "closure != sklearn"

members = [ridge] + [cloudpickle.loads((Path("models/lgbm-blend/components") / f"{t}.pkl").read_bytes()) for t in AUX]
for name, predict, meta in [
    ("ridge-ev", ridge, dict(recipe="eval.py ridge", alpha=1e4, n_features=len(feats), target=target)),
    ("blend3", make_blend(members), dict(members=["ridge-ev", *AUX])),
]:
    out = Path("models") / name
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "model.pkl"
    pkl.write_bytes(cloudpickle.dumps(predict))
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
