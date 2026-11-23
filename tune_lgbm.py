"""Light LightGBM tuning on the walk-forward folds (CPU only, <= 4 configs).

Each fold (eval.walk_forward_folds: exped blocks, 20-exped embargo) early-stops
on an inner hold-out carved from the END of its own training block, behind its
own embargo, so the fold's test block is never seen during fitting or stopping.
Configs are ranked on the mean local event score across folds (a sanity
ranking: the practice board is the source of truth).

Writes data/lgbm_tuning.csv and data/lgbm_best.json (top-2 configs with the
median best iteration), and, with --write-validation, refits the top 2 on all
of train and writes data/lgbm_<name>_validation.parquet.

    .venv/bin/python tune_lgbm.py --write-validation
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import eval as ev

PRIMARY = "target_everest"
BASE = dict(objective="regression", metric="None", learning_rate=0.01, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, seed=7, num_threads=0, verbose=-1)
CONFIGS = {
    "l15_m1000": dict(num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.5),
    "l31_m1000": dict(num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.5),
    "l15_m500": dict(num_leaves=15, min_data_in_leaf=500, feature_fraction=0.5),
    "l31_m2000_d6": dict(num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.5, max_depth=6),
}


def lgbm_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = df[features].to_numpy(dtype=np.float32, copy=True)
    x[x == ev.MISSING] = np.nan
    return x


def exped_corr_feval(expeds: np.ndarray, y: np.ndarray):
    """Mean per-exped Spearman, vectorised (cheap enough to run every iteration)."""
    codes, _ = pd.factorize(expeds)
    y_rank = pd.Series(y).groupby(codes).rank().to_numpy()

    def feval(preds, _data):
        p_rank = pd.Series(preds).groupby(codes).rank().to_numpy()
        df = pd.DataFrame({"g": codes, "p": p_rank, "y": y_rank})
        df["p"] -= df.groupby("g")["p"].transform("mean")
        df["y"] -= df.groupby("g")["y"].transform("mean")
        agg = df.assign(py=df.p * df.y, pp=df.p ** 2, yy=df.y ** 2).groupby("g")[["py", "pp", "yy"]].sum()
        corr = agg.py / np.sqrt(agg.pp * agg.yy).replace(0, np.nan)
        return "exped_corr", float(corr.mean()), True

    return feval


def fit_early_stopped(tr: pd.DataFrame, features, params, inner_holdout: int, embargo: int,
                      max_rounds: int = 4000, patience: int = 300):
    """Early-stop on the last `inner_holdout` expeds of tr (behind an embargo)."""
    order = np.array(sorted(tr["exped"].unique()))
    inner_ho, inner_fit = set(order[-inner_holdout:]), set(order[: -(inner_holdout + embargo)])
    a, h = tr[tr.exped.isin(inner_fit)], tr[tr.exped.isin(inner_ho)]
    dtr = lgb.Dataset(lgbm_matrix(a, features), a[PRIMARY].to_numpy() - 0.5, feature_name=features)
    dho = lgb.Dataset(lgbm_matrix(h, features), h[PRIMARY].to_numpy() - 0.5, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dho],
                        feval=exped_corr_feval(h["exped"].to_numpy(), h[PRIMARY].to_numpy()),
                        callbacks=[lgb.early_stopping(patience, verbose=False)])
    return booster, booster.best_iteration, booster.best_score["valid_0"]["exped_corr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=20)
    ap.add_argument("--inner-holdout", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--write-validation", action="store_true")
    args = ap.parse_args()

    client = ev.EverestAPI()
    b = float(client.explain_scoring()["weights"]["score_multiple_constant"])
    train = ev.load_split("train")
    features = ev.feature_names(train)
    core_avg = ev.core_average(train, features)  # all-feature INOV proxy, as in eval.py

    rows = []
    for k, tr_e, te_e in ev.walk_forward_folds(train["exped"], args.folds, args.embargo):
        tr = train[train.exped.isin(tr_e)]
        te = train[train.exped.isin(te_e)].copy()
        ridge = ev.fit_ridge(tr, features, PRIMARY, args.alpha)
        te["ridge"] = ev.rank_per_exped(te, ridge.predict(ev.design_matrix(te, features)))
        for name, cfg in CONFIGS.items():
            t0 = time.time()
            booster, best, inner = fit_early_stopped(tr, features, {**BASE, **cfg},
                                                     args.inner_holdout, args.embargo)
            te[name] = ev.rank_per_exped(te, booster.predict(lgbm_matrix(te, features),
                                                             num_iteration=best))
            s = ev.summarise(ev.score_per_exped(te, name, PRIMARY, core_avg, b))
            corr_ridge = te.groupby("exped").apply(
                lambda g, n=name: g[n].corr(g["ridge"], method="spearman")).mean()
            rows.append(dict(fold=k, config=name, best_iter=best, inner_corr=inner,
                             corr_ridge=corr_ridge, **s))
            print(f"fold {k} {name:13s} it={best:4d} inner={inner:.4f} fit={s['fit_mean']:.4f} "
                  f"unq={s['unq_mean']:+.4f} event={s['event_score_mean']:+.4f} "
                  f"corr_ridge={corr_ridge:.3f} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(ev.DATA / "lgbm_tuning.csv", index=False)
    summary = res.groupby("config").agg(
        best_iter_median=("best_iter", "median"), fit=("fit_mean", "mean"),
        unq=("unq_mean", "mean"), inov=("inov_mean", "mean"),
        event=("event_score_mean", "mean"), event_std=("event_score_mean", "std"),
        corr_ridge=("corr_ridge", "mean")).sort_values("event", ascending=False)
    print("\n" + summary.round(4).to_string())

    top = list(summary.index[:2])
    best_cfgs = {n: dict(params={**BASE, **CONFIGS[n]},
                         num_boost_round=int(summary.loc[n, "best_iter_median"])) for n in top}
    (ev.DATA / "lgbm_best.json").write_text(json.dumps(best_cfgs, indent=1))
    print("top 2:", top)

    if args.write_validation:
        val = ev.load_split("validation")
        full_x = lgbm_matrix(train, features)
        ridge = ev.fit_ridge(train, features, PRIMARY, args.alpha)
        val["ridge"] = ev.rank_per_exped(val, ridge.predict(ev.design_matrix(val, features)))
        for n in top:
            cfg = best_cfgs[n]
            booster = lgb.train(cfg["params"], lgb.Dataset(full_x, train[PRIMARY] - 0.5,
                                                           feature_name=features),
                                num_boost_round=cfg["num_boost_round"])
            val[n] = ev.rank_per_exped(val, booster.predict(lgbm_matrix(val, features)))
            out = val[[n]].rename(columns={n: "prediction"})
            out.index.name = "id"
            out.to_parquet(ev.DATA / f"lgbm_{n}_validation.parquet")
            c = val.groupby("exped").apply(lambda g: g[n].corr(g["ridge"], method="spearman")).mean()
            s = val.groupby("exped").apply(lambda g: g[n].corr(g[ev.BENCH_COL], method="spearman")).mean()
            print(f"wrote lgbm_{n}_validation.parquet: val corr ridge {c:.4f}, sherpa {s:.4f}")


if __name__ == "__main__":
    main()
