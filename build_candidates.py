"""Build validation predictions for the round-2 candidates (practice board is truth).

  lgbm        LightGBM on all features, -1 as NaN, early-stopped on an embargoed
              hold-out (last --holdout expeds of train, --embargo gap), then refit
              on all of train at the best iteration. CPU only.
  aux3        Ridge per auxiliary target; the 3 whose validation predictions are
              least correlated (|spearman|, per exped) with the target_everest ridge,
              equal-weight rank blend.
  ridge_neut  Ridge fully neutralised against v1_sherpa (orthogonal residual only).

Writes data/<name>_validation.parquet and data/candidates_local.json (local
sanity numbers: hold-out FIT and per-exped correlations on validation).

    .venv/bin/python build_candidates.py
"""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from everestapi import scoring

import eval as ev

PRIMARY = "target_everest"


def spearman_per_exped(df: pd.DataFrame, a: str, b: str) -> float:
    return df.groupby("exped").apply(lambda g: g[a].corr(g[b], method="spearman")).mean()


def holdout_fit(df: pd.DataFrame, col: str) -> float:
    return np.mean([scoring.fit20(g[col], g[PRIMARY]) for _, g in df.groupby("exped")])


def lgbm_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = df[features].to_numpy(dtype=np.float32, copy=True)
    x[x == ev.MISSING] = np.nan
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1000, help="hold-out expeds at the end of train")
    ap.add_argument("--embargo", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1e4)
    args = ap.parse_args()

    train, val = ev.load_split("train"), ev.load_split("validation")
    features = ev.feature_names(train)
    aux_targets = [c for c in train.columns if c.startswith("target_") and c != PRIMARY]

    order = np.array(sorted(train["exped"].unique()))
    ho_expeds = set(order[-args.holdout:])
    fit_expeds = set(order[: -(args.holdout + args.embargo)])
    tr, ho = train[train.exped.isin(fit_expeds)], train[train.exped.isin(ho_expeds)].copy()
    print(f"fit {min(fit_expeds)}..{max(fit_expeds)} | embargo {args.embargo} | "
          f"hold-out {min(ho_expeds)}..{max(ho_expeds)}")
    local = {}

    # ---- plain ridge (reference) -------------------------------------------------
    ridge_ho = ev.fit_ridge(tr, features, PRIMARY, args.alpha)
    ho["ridge"] = ev.rank_per_exped(ho, ridge_ho.predict(ev.design_matrix(ho, features)))
    ridge = ev.fit_ridge(train, features, PRIMARY, args.alpha)
    val["ridge"] = ev.rank_per_exped(val, ridge.predict(ev.design_matrix(val, features)))

    # ---- 1. LightGBM ---------------------------------------------------------------
    params = dict(objective="regression", learning_rate=0.02, num_leaves=31, max_depth=6,
                  min_data_in_leaf=1000, feature_fraction=0.5, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, seed=7, num_threads=0, verbose=-1,
                  metric="None")  # early-stop on the per-exped correlation feval only
    y_tr, y_ho = tr[PRIMARY].to_numpy() - 0.5, ho[PRIMARY].to_numpy() - 0.5
    dtr = lgb.Dataset(lgbm_matrix(tr, features), y_tr, feature_name=features, free_raw_data=False)
    dho = lgb.Dataset(lgbm_matrix(ho, features), y_ho, reference=dtr)
    ho_expeds_arr = ho["exped"].to_numpy()

    def per_exped_corr(preds, data):
        s = pd.DataFrame({"p": preds, "y": data.get_label(), "e": ho_expeds_arr})
        c = s.groupby("e").apply(lambda g: g.p.corr(g.y, method="spearman")).mean()
        return "exped_corr", c, True

    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dho], feval=per_exped_corr,
                        callbacks=[lgb.early_stopping(200, verbose=False),
                                   lgb.log_evaluation(250)])
    best = booster.best_iteration
    ho["lgbm"] = ev.rank_per_exped(ho, booster.predict(lgbm_matrix(ho, features), num_iteration=best))
    print(f"lgbm best_iteration={best}")
    full = lgb.train(params, lgb.Dataset(lgbm_matrix(train, features), train[PRIMARY] - 0.5,
                                         feature_name=features), num_boost_round=best)
    val["lgbm"] = ev.rank_per_exped(val, full.predict(lgbm_matrix(val, features)))
    local["lgbm_best_iteration"] = best

    # ---- 2. auxiliary-target ridges -------------------------------------------------
    tcorr = train[[PRIMARY] + aux_targets].corr(method="spearman")[PRIMARY].drop(PRIMARY)
    aux_corr = {}
    for t in aux_targets:
        m_full = ev.fit_ridge(train, features, t, args.alpha)
        val[f"aux_{t}"] = ev.rank_per_exped(val, m_full.predict(ev.design_matrix(val, features)))
        m_ho = ev.fit_ridge(tr, features, t, args.alpha)
        ho[f"aux_{t}"] = ev.rank_per_exped(ho, m_ho.predict(ev.design_matrix(ho, features)))
        aux_corr[t] = spearman_per_exped(val, f"aux_{t}", "ridge")
    aux_table = pd.DataFrame({"target_corr_train": tcorr, "pred_corr_with_ridge_val": aux_corr})
    aux_table["holdout_fit_on_everest"] = [holdout_fit(ho, f"aux_{t}") for t in aux_table.index]
    aux_table = aux_table.sort_values("pred_corr_with_ridge_val", key=np.abs)
    print("\nauxiliary targets (sorted by |pred corr with everest-ridge| on validation):")
    print(aux_table.round(4).to_string())
    picks = list(aux_table.index[:3])
    print("picked:", picks)
    for df in (val, ho):
        df["aux3"] = df[[f"aux_{t}" for t in picks]].mean(axis=1).groupby(df["exped"]).rank(pct=True)
    local["aux3_picks"] = picks
    local["aux_table"] = aux_table.round(5).to_dict(orient="index")

    # ---- 3. fully neutralised ridge -------------------------------------------------
    for df in (val, ho):
        resid = ev.benchmark_residual(df, "ridge")
        df["ridge_neut"] = resid.groupby(df["exped"]).rank(pct=True)

    # ---- write + local sanity ---------------------------------------------------------
    for name in ["lgbm", "aux3", "ridge_neut"]:
        out = val[[name]].rename(columns={name: "prediction"})
        out.index.name = "id"
        assert out.prediction.notna().all() and out.prediction.between(0, 1).all()
        out.to_parquet(ev.DATA / f"{name}_validation.parquet")
    for name in ["ridge", "lgbm", "aux3", "ridge_neut"]:
        local[name] = dict(
            holdout_fit=holdout_fit(ho, name),
            val_corr_sherpa=spearman_per_exped(val, name, ev.BENCH_COL),
            val_corr_ridge=spearman_per_exped(val, name, "ridge"),
        )
        print(name, {k: round(v, 4) for k, v in local[name].items()})
    (ev.DATA / "candidates_local.json").write_text(json.dumps(local, indent=1, default=float))


if __name__ == "__main__":
    main()
