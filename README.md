# Everesteer hackathon

My scripts from the [Everesteer](https://everesteer.ai) prediction hackathon (November 2026).

## The competition

Everesteer is an agent-first prediction tournament. Each round publishes a cross-section of
instruments with anonymised, binned features. You submit a prediction in [0, 1] for every row,
and it is scored on the forward return that realises later. The event score rewards three
covariances with the target:

- **FIT**: rank correlation with the realised return
- **UNQ**: what is left after removing the benchmark model's direction (weighted 3x)
- **INOV**: what is left after removing a hidden set of core features

The benchmark's own FIT and INOV are subtracted, and the total is squashed with `b·arctan(3S/b)`.
Submissions also have to ship as a cloudpickled `predict()` that runs in the platform sandbox.

## Approach

- **LightGBM** on all published features, with the missing-value sentinel masked to NaN. CPU
  only, with a heavily regularised config (low learning rate, large `min_data_in_leaf`,
  feature and bagging fractions).
- **Walk-forward validation**: folds split by exped (one trading day), never by row, with a
  20-exped embargo between train and test to match the 20-day target horizon. Each fold is
  scored with the SDK's own scoring code (`eval.py`). I used this to choose the round count
  (`cv_rounds.py`), the training window (`cv_window.py`), neutralisation (`cv_neutral.py`)
  and blend membership (`cv_aux3.py`, `swarm.py`).
- **Seed averaging and ensembles**: equal-rank averages of LightGBMs that differ only in seed
  (`build_main_seeds.py`), a 20-model jittered "swarm" (`build_main_swarm.py`), auxiliary-target
  blends (`build_blend.py`, `build_main_targets.py`), and a model distilled from the benchmark's
  predictions (`build_distil*.py`). Early stopping on this dataset was unstable (it picked
  anywhere from 2 to 487 rounds), so averaging was the cheaper way to reduce variance.
- **Round automation**: `round.py` runs a whole round in one command. It checks that the round
  is open, downloads live, predicts with every saved model, submits the staked model first,
  drafts the stake, then submits the rest. `pipeline.py` is the dataset-agnostic
  train → package → validate step. It reads the target, features and missing-value code from
  the schema and never submits.

## Results

| Round | Staked model | Round score | Stake | Payout | Balance |
|---|---|---|---|---|---|
| 1 | distil | 0.080 | $20.00 | +$1.60 | $21.60 |
| 2 | lgbm-main | 0.072 | $21.60 | +$1.55 | $23.15 |
| 3 | main-seeds | 0.070 | $23.15 | +$1.61 | $24.76 |
| 4 | main-seeds | -0.0598 | $24.76 | -$1.48 | $23.28 |

Final: **14th of 63** by balance (+16.4%).

## Files

| File | What it does |
|---|---|
| `pipeline.py` | train → package → validate pipeline (no submit) |
| `round.py` | one-command round: predict, submit, stake draft |
| `eval.py` | offline walk-forward evaluation with event scoring |
| `tune_lgbm.py` | small LightGBM config search on the walk-forward folds |
| `cv_*.py` | walk-forward experiments (rounds, window, neutralisation, aux blend) |
| `swarm.py` | random-mutation LightGBM ensemble experiment |
| `build_*.py` | build and package each submitted model |

Requires `everesteer-api>=0.4.1` and an `EIQ_API_KEY` in the environment. Data and models are
written to `data/` and `models/`, which are git-ignored.

The dataset and the benchmark predictions come from the Everesteer API and are not included.
