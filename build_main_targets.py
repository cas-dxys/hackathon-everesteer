"""main-targets: equal-rank average of 5 LightGBMs with lgbm-main's exact config (DEFAULT_PARAMS incl.
seed, lgbm-main's round count, full event train, target - 0.5): one on target_everest and four on the
auxiliary targets most Spearman-correlated with it on train. No practice board.

    .venv312/bin/python build_main_targets.py [--submit-window round_3 --upload-by 2026-11-22T14:40:00+00:00]
"""
import argparse
import datetime as dt
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

DATA = Path("data/event")
ap = argparse.ArgumentParser()
ap.add_argument("--submit-window", default=None)
ap.add_argument("--upload-by", default=None)
args = ap.parse_args()

main_meta = json.loads(Path("models/lgbm-main/model_meta.json").read_text())
ROUNDS, TARGET, MISSING = main_meta["num_boost_round"], main_meta["target"], main_meta["missing"]
c = EverestAPI()
feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
train = pd.read_parquet(DATA / "pipeline_train.parquet")
feats = [f for f in feats if f in train.columns]
tcols = [t for t in train.columns if t.startswith("target_") and t != TARGET]
corr = train[tcols + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET).sort_values(ascending=False)
AUX = list(corr.index[:4])
log("top aux by spearman with target_everest: " + ", ".join(f"{t} {corr[t]:.3f}" for t in AUX)
    + f" | next: {corr.index[4]} {corr.iloc[4]:.3f}")

x_all = to_matrix(train, feats, MISSING)
members = []
for t in [TARGET, *AUX]:
    keep = train[t].notna().to_numpy()
    center = 0.5 if train.loc[keep, t].between(0, 1).all() else float(train.loc[keep, t].mean())
    b = lgb.train(DEFAULT_PARAMS, lgb.Dataset(x_all[keep], train[t].to_numpy()[keep] - center), num_boost_round=ROUNDS)
    members.append(make_predict(b.model_to_string(), feats, MISSING))
    log(f"{t}: {keep.sum()} rows, {ROUNDS} rounds")

out = Path("models/main-targets")
out.mkdir(parents=True, exist_ok=True)
pkl = out / "model.pkl"
pkl.write_bytes(cloudpickle.dumps(make_blend(members)))
save_components(out, [TARGET, *AUX], members)
(out / "model_meta.json").write_text(json.dumps(dict(members={t: ROUNDS for t in [TARGET, *AUX]},
                                                     aux_spearman={t: float(corr[t]) for t in AUX}, params=DEFAULT_PARAMS,
                                                     num_boost_round=ROUNDS, python_version="3.12",
                                                     sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))
m = c.create_model(name="main-targets")
(out / "model_id.json").write_text(json.dumps({"main-targets": m["id"]}))
log(f"model main-targets id={m['id']} public name="
    f"{next(x['name'] for x in c.get_models()['models'] if x['id'] == m['id'])}")

if args.submit_window:
    w = args.submit_window
    live = pd.read_parquet(DATA / f"round_{w}_live.parquet")
    pred = cloudpickle.loads(pkl.read_bytes())(live)
    col = pred.iloc[:, 0]
    assert pred.index.equals(live.index) and np.isfinite(col).all() and col.between(0, 1).all()
    ms = cloudpickle.loads(Path("models/lgbm-main/model.pkl").read_bytes())(live).iloc[:, 0]
    log(f"{w} live {len(live)} rows: main-targets vs lgbm-main spearman {col.corr(ms, method='spearman'):.3f}")
    path = DATA / f"round_{w}_live_main-targets.parquet"
    pd.DataFrame({"id": pred.index.astype(str), "prediction": col.to_numpy()}).to_parquet(path, index=False)
    now = dt.datetime.now(dt.timezone.utc)
    if args.upload_by and now > dt.datetime.fromisoformat(args.upload_by):
        log(f"ABANDON: past {args.upload_by}; main-targets NOT submitted"); raise SystemExit(1)
    cad = c.get_status().get("cadence") or {}
    if cad.get("open_window") != w or cad.get("intake_fenced") is not False:
        log(f"ABANDON: {w} not submittable (open_window={cad.get('open_window')}, fenced={cad.get('intake_fenced')})")
        raise SystemExit(1)
    log(f"uploading main-targets to {w}")
    try:
        r = c.submit_event_predictions(m["id"], str(path), model_pkl=str(pkl), model_pkl_python_version="3.12",
                                       client_label=f"{w} manual", wait=True, timeout=240)
        log("main-targets: " + json.dumps({k: r.get(k) for k in ["upload_id", "status", "error"]}, default=str))
    except Exception as e:
        log(f"main-targets: ERROR {type(e).__name__}: {str(e)[:400]}")
