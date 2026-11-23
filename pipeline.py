"""Dataset-agnostic train -> package -> validate pipeline (never submits).

Works on whatever dataset the API key in EIQ_API_KEY serves (tournament or a
hackathon event). Everything dataset-specific is read from the platform:

  * get_started         mode, supported model Python versions
  * get_dataset_schema  primary_target, the feature list, the missing-value code,
                        which splits are labelled

Then it downloads every labelled split plus `live`, picks the boosting rounds by
early stopping on an embargoed tail of the labelled data, refits on ALL labelled
rows, packages a cloudpickled predict(live_features) -> single-column DataFrame
in [0, 1], smoke-tests it on the live rows, and runs validate_submission on the
live predictions. It never calls a submit endpoint.

Run it under an interpreter the sandbox accepts, with that lane's library pins
installed (see TOMORROW.md), e.g.

    .venv312/bin/python pipeline.py
    .venv312/bin/python pipeline.py --config data/lgbm_best.json --out models/event
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from everestapi import EverestAPI
from everestapi.client import EverestError

ERA = "exped"  # the platform's time-unit column on every dataset it serves
# The practice-board winner of tune_lgbm.py (l31_m2000_d6). The round count is NOT
# taken from here: it is re-picked by early stopping on whatever dataset is served.
DEFAULT_PARAMS = dict(objective="regression", metric="None", learning_rate=0.01,
                      num_leaves=31, max_depth=6, min_data_in_leaf=2000, feature_fraction=0.5,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=7,
                      num_threads=0, verbose=-1)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- schema


def read_schema(client: EverestAPI) -> dict:
    schema = client.get_dataset_schema()  # target, encoding, splits
    # The verbose call carries ONLY feature_sets (as member lists) and targets.
    sets = client.get_dataset_schema(verbose=True)["feature_sets"]
    # Prefer the full published set; otherwise the largest one the dataset declares.
    name = "all" if sets.get("all") else max(sets, key=lambda k: len(sets[k] or []))
    encoding = schema.get("feature_encoding") or {}
    # Schema contract: -1 is the missing code on every dataset; the encoding block
    # restates it when present and may be null on trees that declare none.
    missing = encoding.get("missing", -1)
    labelled = [s for s, meta in (schema.get("splits") or {}).items()
                if meta.get("labeled") and s != "live"] or ["train"]
    return dict(primary_target=schema["primary_target"], feature_set=name,
                features=list(sets[name]), missing=missing, encoding=encoding,
                labelled_splits=labelled, version=schema.get("version"))


def python_lane(started: dict) -> str:
    mine = f"{sys.version_info.major}.{sys.version_info.minor}"
    supported = (started.get("model_python_versions") or {}).get("supported") or []
    if supported and mine not in supported:
        raise SystemExit(f"Python {mine} is not an accepted model lane {supported}: the .pkl "
                         f"would be refused at upload. Re-run under one of those (TOMORROW.md).")
    return mine


# ----------------------------------------------------------------------------- data


def download(client: EverestAPI, split: str, out_dir: Path) -> pd.DataFrame | None:
    path = out_dir / f"pipeline_{split}.parquet"
    try:
        client.download_dataset(split=split, output_path=str(path))
    except EverestError as e:
        log(f"download {split}: unavailable ({e})")
        return None
    return pd.read_parquet(path)


def to_matrix(df: pd.DataFrame, features: list[str], missing) -> np.ndarray:
    x = df.reindex(columns=features).to_numpy(dtype=np.float32, copy=True)
    if missing is not None:
        x[x == missing] = np.nan
    return x


# ----------------------------------------------------------------------------- train


def exped_corr_feval(expeds: np.ndarray, y: np.ndarray):
    codes, _ = pd.factorize(expeds)
    y_rank = pd.Series(y).groupby(codes).rank().to_numpy()

    def feval(preds, _data):
        df = pd.DataFrame({"g": codes, "p": pd.Series(preds).groupby(codes).rank().to_numpy(),
                           "y": y_rank})
        df["p"] -= df.groupby("g")["p"].transform("mean")
        df["y"] -= df.groupby("g")["y"].transform("mean")
        agg = df.assign(py=df.p * df.y, pp=df.p ** 2, yy=df.y ** 2).groupby("g")[["py", "pp", "yy"]].sum()
        return "exped_corr", float((agg.py / np.sqrt(agg.pp * agg.yy).replace(0, np.nan)).mean()), True

    return feval


def pick_rounds(df, features, target, missing, params, holdout_frac, embargo) -> tuple[int, float]:
    order = np.array(sorted(df[ERA].unique()))
    n_ho = max(int(len(order) * holdout_frac), 20)
    ho_e, fit_e = set(order[-n_ho:]), set(order[: -(n_ho + embargo)])
    a, h = df[df[ERA].isin(fit_e)], df[df[ERA].isin(ho_e)]
    log(f"early stopping: fit {len(fit_e)} {ERA}s | embargo {embargo} | hold-out {len(ho_e)}")
    center = 0.5 if df[target].between(0, 1).all() else float(df[target].mean())
    dtr = lgb.Dataset(to_matrix(a, features, missing), a[target].to_numpy() - center)
    dho = lgb.Dataset(to_matrix(h, features, missing), h[target].to_numpy() - center, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dho],
                        feval=exped_corr_feval(h[ERA].to_numpy(), h[target].to_numpy()),
                        callbacks=[lgb.early_stopping(300, verbose=False)])
    return booster.best_iteration, booster.best_score["valid_0"]["exped_corr"]


# ----------------------------------------------------------------------------- package


def make_predict(model_str: str, features: list[str], missing):
    """The uploaded artifact: a plain closure over the booster text and the
    preprocessing, so nothing from this repo needs to be importable in the sandbox."""

    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        import lightgbm as lgb
        import numpy as np
        import pandas as pd

        booster = lgb.Booster(model_str=model_str)
        x = live_features.reindex(columns=features).to_numpy(dtype=np.float32, copy=True)
        if missing is not None:
            x[x == missing] = np.nan
        raw = pd.Series(booster.predict(x), index=live_features.index)
        return raw.rank(pct=True).clip(0.0, 1.0).to_frame("prediction")

    return predict


def main() -> None:
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None,
                    help="tune_lgbm.py's lgbm_best.json, to override DEFAULT_PARAMS")
    ap.add_argument("--config-name", default=None, help="entry in --config (default: first)")
    ap.add_argument("--out", type=Path, default=Path("models/pipeline"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--embargo", type=int, default=20, help="gap in eras before the hold-out")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--rounds", type=int, default=None,
                    help="fixed boosting rounds; skips early stopping (it is unstable on Atlas)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    client = EverestAPI()
    started = client.get_started()
    lane = python_lane(started)
    mode = started.get("mode")
    log(f"mode={mode} python lane={lane}")
    if started.get("welcome"):
        print("\n--- welcome (verbatim) ---\n" + started["welcome"] + "\n---\n")

    sch = read_schema(client)
    target, features, missing = sch["primary_target"], sch["features"], sch["missing"]
    log(f"dataset={sch['version']} target={target} features={len(features)} "
        f"(set '{sch['feature_set']}') missing={missing} labelled={sch['labelled_splits']}")

    params = dict(DEFAULT_PARAMS)
    if args.config and args.config.exists():
        cfgs = json.loads(args.config.read_text())
        name = args.config_name or next(iter(cfgs))
        params = cfgs[name]["params"]
        log(f"params from {args.config} [{name}]")

    frames = [f for s in sch["labelled_splits"] if (f := download(client, s, args.data_dir)) is not None]
    labelled = pd.concat(frames)
    labelled = labelled[labelled[target].notna()]
    features = [f for f in features if f in labelled.columns]
    log(f"labelled rows {len(labelled)} across {labelled[ERA].nunique()} {ERA}s")

    live = download(client, "live", args.data_dir)

    if args.rounds:
        best_iter, ho_corr = args.rounds, float("nan")
    else:
        best_iter, ho_corr = pick_rounds(labelled, features, target, missing, params,
                                         args.holdout_frac, args.embargo)
    log(f"best_iteration={best_iter} hold-out per-{ERA} corr={ho_corr:.4f}; refitting on all labelled")
    center = 0.5 if labelled[target].between(0, 1).all() else float(labelled[target].mean())
    booster = lgb.train(params, lgb.Dataset(to_matrix(labelled, features, missing),
                                            labelled[target].to_numpy() - center),
                        num_boost_round=best_iter)

    predict = make_predict(booster.model_to_string(), features, missing)
    pkl_path = args.out / "model.pkl"
    pkl_path.write_bytes(cloudpickle.dumps(predict))
    sha = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
    meta = dict(python_version=lane, sha256=sha, size_bytes=pkl_path.stat().st_size,
                dataset=sch["version"], target=target, n_features=len(features),
                missing=missing, params=params, num_boost_round=best_iter,
                holdout_corr=ho_corr, mode=mode)
    (args.out / "model_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {pkl_path} ({meta['size_bytes']} bytes, sha256 {sha[:12]}…, python_version={lane})")

    if live is None:
        log("no live split served right now: skipping the smoke test on live and validate_submission")
    else:
        reloaded = cloudpickle.loads(pkl_path.read_bytes())
        feats_only = live[[c for c in live.columns if c in features]]
        out = reloaded(feats_only)
        assert isinstance(out, pd.DataFrame) and out.shape[1] == 1, "predict must return 1 column"
        assert out.index.equals(live.index), "predict must keep the live id index"
        assert out.iloc[:, 0].between(0, 1).all() and np.isfinite(out.iloc[:, 0]).all()
        out.index.name = "id"
        out.to_parquet(args.out / "live_predictions.parquet")
        exped = str(live[ERA].iloc[0]) if ERA in live.columns else None
        log(f"smoke test ok on {len(out)} live rows ({ERA} {exped})")
        try:
            v = client.validate_submission({str(k): float(x) for k, x in out.iloc[:, 0].items()},
                                           exped=exped)
            print(json.dumps(v, indent=1, default=str))
        except EverestError as e:
            log(f"validate_submission refused: {e}")

    log(f"NOT submitted. total runtime {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
