"""Offline evaluation harness for Everesteer EVENT scoring.

Walk-forward folds split by exped (never by row) with an embargo between the
train and test blocks, scored per exped with the SDK's own scoring toolkit
(everestapi.scoring): FIT, UNQ and INOV with the event rules (zero rule off),
and the event round score (unq_zr / inov_zr are the zero-rule-on tournament
variants, which is what the practice board reports)

    S     = FIT + 3*UNQ + INOV - (benchmark FIT + benchmark INOV)
    score = b * arctan(3 * S / b)

with b read live from explain_scoring. The tournament score
b*arctan((FIT + UNQ + INOV)/b), zero rule on, is reported alongside.

INOV caveat: the platform's core feature set (10 features) is deliberately not
published, so by default INOV is measured against a PROXY, the equal-weight
average of every feature. Pass --core-features <file> (one name per line) to
use a real list.

Usage:
    .venv/bin/python eval.py                        # benchmark + ridge on 5 folds
    .venv/bin/python eval.py --write-validation     # also write ridge validation preds
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from everestapi import EverestAPI, scoring
from sklearn.linear_model import Ridge

DATA = Path(__file__).parent / "data"
MISSING = -1
BENCH_COL = "v1_sherpa"


# --------------------------------------------------------------------------- data


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"{split}.parquet")
    bench = pd.read_parquet(DATA / f"bench_{split}.parquet")[BENCH_COL]
    df[BENCH_COL] = bench.reindex(df.index)
    return df.sort_values("exped", kind="stable")


def feature_names(df: pd.DataFrame) -> list[str]:
    """The published `all` set, falling back to the file's feature columns offline."""
    try:
        names = EverestAPI().get_dataset_schema(verbose=True)["feature_sets"]["all"]
        return [n for n in names if n in df.columns]
    except Exception:
        return [c for c in df.columns if c.startswith("feature_")]


def design_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Bins 0..9 centred on the mid-bin; -1 is missing and lands on 0 (the centre),
    never as an ordinal below bin 0."""
    x = df[features].to_numpy(dtype=np.float32)
    x = (x - 4.5) / 9.0
    x[df[features].to_numpy() == MISSING] = 0.0
    return x


def core_average(df: pd.DataFrame, core: list[str]) -> pd.Series:
    """Per-row equal-weight average of the core features (-1 left out), per the SDK."""
    return scoring.core_feature_average(df[core])


# --------------------------------------------------------------------------- folds


def walk_forward_folds(expeds: pd.Series, n_folds: int = 5, embargo: int = 20):
    """Expanding-window folds on the exped axis.

    The ordered exped list is cut into n_folds + 1 contiguous blocks; fold k tests
    on block k (k = 1..n_folds) and trains on every exped strictly before it,
    minus the last `embargo` expeds (the 20-day target horizon), so no training
    label overlaps the test period. Yields (fold, train_expeds, test_expeds).
    """
    order = np.array(sorted(expeds.unique()))
    blocks = np.array_split(order, n_folds + 1)
    for k in range(1, n_folds + 1):
        test = blocks[k]
        first_test = np.searchsorted(order, test[0])
        train = order[: max(first_test - embargo, 0)]
        yield k, set(train), set(test)


# --------------------------------------------------------------------------- scoring


def score_per_exped(
    df: pd.DataFrame,
    pred_col: str,
    target: str,
    core_avg: pd.Series,
    b: float,
) -> pd.DataFrame:
    """Per-exped FIT / UNQ / INOV and the event + tournament round scores."""
    rows = []
    for exped, g in df.groupby("exped", sort=True):
        p, y, bench, avg = g[pred_col], g[target], g[BENCH_COL], core_avg.loc[g.index]
        fit = scoring.fit20(p, y)
        unq = scoring.unq20(p, bench, y, zero_rule=False)
        inov = scoring.inov(p, avg.to_frame(), y, zero_rule=False)
        bench_fit = scoring.fit20(bench, y)
        bench_inov = scoring.inov(bench, avg.to_frame(), y, zero_rule=False)
        event = scoring.event_score(
            fit, unq, inov,
            score_multiple_constant=b,
            benchmark_fit=bench_fit,
            benchmark_inov=bench_inov,
        )
        unq_zr = scoring.unq20(p, bench, y)
        inov_zr = scoring.inov(p, avg.to_frame(), y)
        tourn = scoring.payout(fit, unq_zr, inov=inov_zr, score_multiple_constant=b)
        bench_corr = p.corr(bench, method="spearman")
        rows.append(dict(exped=exped, fit=fit, unq=unq, inov=inov, unq_zr=unq_zr,
                         inov_zr=inov_zr, event_score=event,
                         tourn_score=tourn, bench_corr=bench_corr))
    return pd.DataFrame(rows).set_index("exped")


def summarise(per_exped: pd.DataFrame) -> dict:
    out = {}
    for col in ["fit", "unq", "inov", "unq_zr", "inov_zr", "event_score", "tourn_score"]:
        out[f"{col}_mean"] = per_exped[col].mean()
        out[f"{col}_std"] = per_exped[col].std()
    out["bench_corr"] = per_exped["bench_corr"].mean()
    out["n_expeds"] = len(per_exped)
    return out


# --------------------------------------------------------------------------- models


def fit_ridge(df: pd.DataFrame, features: list[str], target: str, alpha: float) -> Ridge:
    y = df[target].to_numpy(dtype=np.float64) - 0.5
    keep = ~np.isnan(y)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design_matrix(df, features)[keep], y[keep])
    return model


def rank_per_exped(df: pd.DataFrame, raw: np.ndarray) -> pd.Series:
    """Submission shape: values in [0, 1], ranked within each exped."""
    s = pd.Series(raw, index=df.index)
    return s.groupby(df["exped"]).rank(pct=True)


def gaussianize_per_exped(df: pd.DataFrame, col: str) -> pd.Series:
    """Rank -> clip -> inverse normal within each exped, as the scorer does."""
    return df.groupby("exped")[col].transform(
        lambda s: pd.Series(scoring._rank_and_gaussianize(s), index=s.index))


def benchmark_residual(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """pred_col with the benchmark's direction projected out per exped, on
    gaussianized ranks: the same projection UNQ uses (100% neutralisation)."""
    z = pd.DataFrame({"exped": df["exped"],
                      "p": gaussianize_per_exped(df, pred_col),
                      "m": gaussianize_per_exped(df, BENCH_COL)})
    beta = (z.assign(pm=z.p * z.m, mm=z.m * z.m).groupby("exped")[["pm", "mm"]]
            .transform("sum").pipe(lambda t: t.pm / t.mm.where(t.mm != 0)))
    return z.p - beta.fillna(0.0) * z.m


def benchmark_residual_blend(df: pd.DataFrame, pred_col: str, lam: float) -> pd.Series:
    """rank(benchmark) + lam * rank(residual), re-ranked to [0, 1] per exped."""
    resid = benchmark_residual(df, pred_col)
    rank = lambda s: s.groupby(df["exped"]).rank(pct=True)  # noqa: E731
    return rank(rank(df[BENCH_COL]) + lam * rank(resid))


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--core-features", type=Path, default=None,
                    help="file with one core feature name per line (default: all-feature proxy)")
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.1, 0.25, 0.5, 1.0],
                    help="residual weights for the benchmark + ridge-residual blend")
    ap.add_argument("--write-validation", action="store_true",
                    help="fit ridge on all of train and write validation prediction files")
    ap.add_argument("--write-lambdas", type=float, nargs="*", default=[],
                    help="blend lambdas to write validation files for (with --write-validation)")
    args = ap.parse_args()

    client = EverestAPI()
    scoring_cfg = client.explain_scoring()
    b = float(scoring_cfg["weights"]["score_multiple_constant"])
    target = client.get_dataset_schema()["primary_target"]

    train = load_split("train")
    features = feature_names(train)
    core = (args.core_features.read_text().split() if args.core_features else features)
    core_avg = core_average(train, core)
    print(f"b={b:.5f} target={target} features={len(features)} "
          f"core={'file' if args.core_features else 'PROXY(all)'}:{len(core)} "
          f"alpha={args.alpha:g} embargo={args.embargo}")

    results = []
    for k, tr_e, te_e in walk_forward_folds(train["exped"], args.folds, args.embargo):
        tr = train[train["exped"].isin(tr_e)]
        te = train[train["exped"].isin(te_e)].copy()
        print(f"fold {k}: train {min(tr_e)}..{max(tr_e)} ({len(tr_e)} expeds) | "
              f"test {min(te_e)}..{max(te_e)} ({len(te_e)} expeds)")

        te["benchmark"] = te[BENCH_COL]
        model = fit_ridge(tr, features, target, args.alpha)
        te["ridge"] = rank_per_exped(te, model.predict(design_matrix(te, features)))

        names = ["benchmark", "ridge"]
        for lam in args.lambdas:
            te[f"blend_{lam:g}"] = benchmark_residual_blend(te, "ridge", lam)
            names.append(f"blend_{lam:g}")

        for name in names:
            per = score_per_exped(te, name, target, core_avg, b)
            results.append(dict(fold=k, model=name, **summarise(per)))

    res = pd.DataFrame(results)
    pd.set_option("display.width", 200, "display.float_format", "{:.5f}".format)
    cols = ["fold", "model", "fit_mean", "fit_std", "unq_mean", "unq_std", "inov_mean",
            "inov_std", "unq_zr_mean", "inov_zr_mean", "event_score_mean", "event_score_std",
            "tourn_score_mean", "bench_corr"]
    print("\nPer fold (mean/std across the fold's expeds):")
    print(res[cols].sort_values(["model", "fold"]).to_string(index=False))
    print("\nAcross folds (mean of fold means, std of fold means):")
    agg = res.groupby("model")[[c for c in cols if c.endswith("_mean")] + ["bench_corr"]]
    print(pd.concat({"mean": agg.mean(), "std": agg.std()}, axis=1).T.to_string())
    res.to_csv(DATA / "eval_folds.csv", index=False)

    if args.write_validation:
        val = load_split("validation")
        model = fit_ridge(train, features, target, args.alpha)
        val["ridge"] = rank_per_exped(val, model.predict(design_matrix(val, features)))
        val["benchmark"] = val[BENCH_COL]
        names = ["ridge", "benchmark"]
        for lam in args.write_lambdas:
            val[f"blend_{lam:g}"] = benchmark_residual_blend(val, "ridge", lam)
            names.append(f"blend_{lam:g}")
        print()
        for name in names:
            out = val[[name]].rename(columns={name: "prediction"})
            out.index.name = "id"
            path = DATA / f"{name}_validation.parquet"
            out.to_parquet(path)
            corr = val.groupby("exped").apply(
                lambda g, n=name: g[n].corr(g[BENCH_COL], method="spearman")).mean()
            print(f"wrote {path} ({len(out)} rows); "
                  f"mean per-exped spearman with benchmark: {corr:.4f}")


if __name__ == "__main__":
    main()
