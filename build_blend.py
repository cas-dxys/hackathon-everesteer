"""Equal-rank blend of lgbm-main plus LightGBMs on auxiliary targets.

    .venv312/bin/python build_blend.py --name lgbm-blend \
        --aux target_Azegza target_Tamalout target_Aday
    .venv312/bin/python build_blend.py --name lgbm-blend --components-only   # re-save from model.pkl

Each auxiliary model picks its rounds by early stopping (pipeline.pick_rounds) on
an embargoed tail, then refits on every labelled row of that target. Writes, under
models/<name>/:

  model.pkl                 the blend: predict(live_features) -> 1-column DataFrame in [0, 1]
  components/<member>.pkl   each member's own predict(), same contract
  components/<member>.txt   each member's LightGBM booster text
  model_meta.json, model_id.json

then scores the blend on the practice board (validation lane: free, never staked).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cloudpickle
import lightgbm as lgb
import pandas as pd
from everestapi import EverestAPI

from pipeline import DEFAULT_PARAMS, log, make_predict, pick_rounds, to_matrix

DATA_DIR = Path("data/event")
MAIN = Path("models/lgbm-main/model.pkl")


def make_blend(members):
    def predict(live_features):
        import pandas as pd

        cols = [m(live_features).iloc[:, 0] for m in members]
        return pd.concat(cols, axis=1).mean(axis=1).rank(pct=True).clip(0.0, 1.0).to_frame("prediction")

    return predict


def booster_text(member) -> str:
    """The model_str a pipeline.make_predict closure was built over."""
    return next(c.cell_contents for c in member.__closure__ if isinstance(c.cell_contents, str))


def save_components(out: Path, names: list[str], members: list) -> None:
    comp = out / "components"
    comp.mkdir(parents=True, exist_ok=True)
    for name, m in zip(names, members):
        (comp / f"{name}.pkl").write_bytes(cloudpickle.dumps(m))
        (comp / f"{name}.txt").write_text(booster_text(m))
    log(f"saved {len(members)} components under {comp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--aux", nargs="+", default=[])
    ap.add_argument("--components-only", action="store_true",
                    help="split an existing models/<name>/model.pkl into components; no training")
    args = ap.parse_args()
    out = Path("models") / args.name
    out.mkdir(parents=True, exist_ok=True)

    if args.components_only:
        blend = cloudpickle.loads((out / "model.pkl").read_bytes())
        members = blend.__closure__[0].cell_contents
        names = ["lgbm-main"] + list(json.loads((out / "model_meta.json").read_text())["members"])[1:]
        save_components(out, names, members)
        return

    c = EverestAPI()
    feats = list(c.get_dataset_schema(verbose=True)["feature_sets"]["all"])
    df = pd.read_parquet(DATA_DIR / "pipeline_train.parquet")
    feats = [f for f in feats if f in df.columns]
    members = [cloudpickle.loads(MAIN.read_bytes())]
    info = {"lgbm-main": json.loads(MAIN.with_name("model_meta.json").read_text())["num_boost_round"]}
    for t in args.aux:
        d = df[df[t].notna()]
        n, corr = pick_rounds(d, feats, t, -1, DEFAULT_PARAMS, 0.15, 20)
        log(f"{t}: rounds={n} holdout corr={corr:.4f}")
        center = 0.5 if d[t].between(0, 1).all() else float(d[t].mean())
        b = lgb.train(DEFAULT_PARAMS, lgb.Dataset(to_matrix(d, feats, -1), d[t].to_numpy() - center),
                      num_boost_round=n)
        members.append(make_predict(b.model_to_string(), feats, -1))
        info[t] = n
    save_components(out, list(info), members)

    pkl = out / "model.pkl"
    pkl.write_bytes(cloudpickle.dumps(make_blend(members)))
    v = pd.read_parquet(DATA_DIR / "validation.parquet")
    pred = cloudpickle.loads(pkl.read_bytes())(v)
    assert pred.index.equals(v.index) and pred.iloc[:, 0].between(0, 1).all()
    main_pred = members[0](v).iloc[:, 0]
    log(f"blend vs lgbm-main spearman on validation: {pred.iloc[:, 0].corr(main_pred, method='spearman'):.3f}")
    val_path = DATA_DIR / f"{args.name.replace('-', '_')}_validation.parquet"
    pd.DataFrame({"id": pred.index.astype(str), "prediction": pred.iloc[:, 0].to_numpy()}).to_parquet(val_path, index=False)
    (out / "model_meta.json").write_text(json.dumps(dict(
        members=info, python_version="3.12", sha256=hashlib.sha256(pkl.read_bytes()).hexdigest()), indent=1))

    m = c.create_model(name=args.name)
    (out / "model_id.json").write_text(json.dumps({args.name: m["id"]}))
    r = c.submit_validation_diagnostics(m["id"], str(val_path), model_pkl=str(pkl), model_pkl_python_version="3.12",
                                        client_label=f"{args.name} practice", wait=True, timeout=600)
    print(json.dumps({k: r.get(k) for k in ["upload_id", "status", "error", "fit20", "unq", "inov", "round_score"]},
                     default=str))


if __name__ == "__main__":
    main()
