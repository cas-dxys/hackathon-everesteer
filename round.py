"""One-command event round: status -> live -> predict -> submit stake model -> stake draft -> submit the rest.

    .venv312/bin/python round.py                                   # submit only, no stake draft
    .venv312/bin/python round.py --stake-model distil --amount 5
    .venv312/bin/python round.py --stake-model distil --only       # submit + stake distil, nothing else
    .venv312/bin/python round.py --dry-run                         # every read, no submit, no draft

Steps, each printed:

  1. get_status; abort if cadence.intake_fenced (no open round to submit to)
  2. download_dataset(split="live") for the open round
  3. predict it with each saved pkl (SUBMIT_MODELS)
  4. submit --stake-model FIRST via submit_event_predictions
  5. as soon as that upload succeeds: get_event_staking and, if draft_window names the
     open round, draft set_stake_allocation(<stake model's ID>, amount_usdc=--amount,
     default the full max_stakeable balance)
  6. submit every other model, one at a time

The Python SDK has no submit_event_predictions_batch, so each model is its own
submit_event_predictions call with its own outcome. Each one spends an upload from
the event pool. Every step from 3 on is wrapped: an error is printed and the
remaining steps still run. Steps 1-2 are hard dependencies (no confirmed open round,
or no rows to predict, means there is nothing safe to do) and abort.

A draft locks when the round closes and can't be undone after that. --dry-run
never writes anything: when the round is fenced it carries on with the
validation rows standing in for live, so the predict path is still exercised.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from decimal import Decimal
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from everestapi import EverestAPI

SUBMIT_MODELS = ["lgbm-main", "distil", "main-distil", "main-seeds", "distil-seeds", "main-swarm",
                 "main-targets"]  # submitted every round
DATA_DIR = Path("data/event")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_error(step: str, e: Exception) -> None:
    log(f"   ERROR in {step}: {type(e).__name__}: {e}")
    print(traceback.format_exc(limit=3), flush=True)


def micro_to_usdc(micro: int) -> str:
    return str(Decimal(int(micro)) / Decimal(1_000_000))


def load_model(name: str) -> dict:
    d = Path("models") / name
    meta = json.loads((d / "model_meta.json").read_text())
    return dict(name=name, dir=d, id=json.loads((d / "model_id.json").read_text())[name],
                python_version=meta["python_version"], sha256=meta["sha256"],
                predict=cloudpickle.loads((d / "model.pkl").read_bytes()))


def submit(client: EverestAPI, m: dict, window: str | None, dry: bool) -> bool:
    """One model's upload to the round lane. True only when it landed."""
    if "pred_path" not in m:
        log(f"   skip submit {m['name']}: no predictions (load or predict failed)")
        return False
    if dry:
        log(f"   (dry run) would call submit_event_predictions({m['name']}={m['id']}, {m['pred_path']}, model_pkl=...)")
        return True
    try:
        r = client.submit_event_predictions(
            m["id"], str(m["pred_path"]), model_pkl=str(m["dir"] / "model.pkl"),
            model_pkl_python_version=m["python_version"], client_label=f"round.py {window}", wait=True)
    except Exception as e:
        log_error(f"submit {m['name']}", e)
        return False
    ok = not r.get("error") and r.get("status") not in ("failed", "error")
    log(f"   submit {m['name']}: {'OK' if ok else 'FAILED'} " + json.dumps(
        {k: r.get(k) for k in ["upload_id", "status", "error", "uploads_remaining"]}, default=str))
    return ok


def draft_stake(client: EverestAPI, m: dict, amount_arg: str | None, window: str | None, dry: bool) -> None:
    s = client.get_event_staking()
    draft_window = s.get("draft_window")
    log(f"   get_event_staking: staking_armed={s.get('staking_armed')} stake_window_open={s.get('stake_window_open')} "
        f"draft_window={draft_window} max_stakeable_micro={s.get('max_stakeable_micro')} "
        f"unavailable={s.get('max_stakeable_unavailable')} min_stake_micro={s.get('min_stake_micro')}")
    if amount_arg is not None:
        amount = amount_arg
    elif s.get("max_stakeable_unavailable") or s.get("max_stakeable_micro") is None:
        log("   no draft: balance unreadable right now")
        return
    else:
        amount = micro_to_usdc(s["max_stakeable_micro"])
    if not window or draft_window != window:
        log(f"   no draft: draft_window ({draft_window}) does not name the open round ({window}); "
            f"would have drafted {amount} USDC on {m['name']} ({m['id']})")
        return
    if dry:
        log(f"   (dry run) would call set_stake_allocation({m['id']!r} [{m['name']}], amount_usdc={amount!r}, "
            f"window={window!r})")
        return
    a = client.set_stake_allocation(m["id"], amount_usdc=amount, window=window)
    log(f"   set_stake_allocation {m['name']} ({m['id']}): " + json.dumps(a, default=str)[:800])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake-model", default=None, choices=SUBMIT_MODELS,
                    help="model to submit first and draft the stake on (default: none, no stake draft)")
    ap.add_argument("--amount", default=None,
                    help="USDC as a decimal string, e.g. '5' or '2.5' (default: full max_stakeable balance)")
    ap.add_argument("--only", action="store_true",
                    help="submit ONLY --stake-model (plus its stake draft); skip every other model")
    ap.add_argument("--dry-run", action="store_true", help="reads only: no submit, no stake draft")
    args = ap.parse_args()
    if args.only and not args.stake_model:
        ap.error("--only needs --stake-model (it submits only that model)")
    dry = args.dry_run
    if args.amount is not None:
        Decimal(args.amount)  # refuse a malformed amount before anything is submitted
    client = EverestAPI()

    # Stake model first, then the rest in SUBMIT_MODELS order.
    if args.only:
        order = [args.stake_model]
    else:
        order = ([args.stake_model] if args.stake_model else []) + [n for n in SUBMIT_MODELS if n != args.stake_model]
    models = []
    for n in order:
        try:
            m = load_model(n)
            models.append(m)
            log(f"{'DRY RUN, ' if dry else ''}model {m['name']} id={m['id']} pkl sha256={m['sha256'][:12]}…")
        except Exception as e:
            log_error(f"load {n}", e)
            models.append(dict(name=n, id=None))
    stake_m = models[0] if args.stake_model else None
    if stake_m:
        log(f"stake model={stake_m['name']} id={stake_m['id']} amount={args.amount or 'full balance'}"
            + (" | --only: no other model is submitted" if args.only else ""))
    else:
        log("no --stake-model: submit only, no stake draft")

    # 1. status (hard dependency)
    try:
        status = client.get_status()
    except Exception as e:
        log_error("get_status", e)
        log("ABORT: can't confirm a round is open. Nothing submitted, nothing staked.")
        raise SystemExit(1)
    cad = status.get("cadence") or {}
    window = cad.get("open_window")
    log(f"1 get_status: mode={status.get('mode')} phase={cad.get('phase')} open_window={window} "
        f"intake_fenced={cad.get('intake_fenced')} phase_ends_at={cad.get('phase_ends_at')}")
    fenced = cad.get("intake_fenced") is not False or not window
    if fenced and not dry:
        log("ABORT: intake is fenced (no open round). Nothing submitted, nothing staked.")
        raise SystemExit(1)
    if fenced:
        log("   (dry run) intake fenced: a real run aborts here; continuing on validation rows")

    # 2. live (hard dependency)
    split = "validation" if fenced else "live"
    path = DATA_DIR / f"round_{window or 'none'}_{split}.parquet"
    try:
        client.download_dataset(split=split, output_path=str(path))
        live = pd.read_parquet(path)
    except Exception as e:
        log_error(f"download {split}", e)
        log("ABORT: no rows to predict. Nothing submitted, nothing staked.")
        raise SystemExit(1)
    log(f"2 download {split}: {len(live)} rows, {live['exped'].nunique()} expeds -> {path}")

    # 3. predict
    for m in models:
        if m.get("id") is None:
            continue
        try:
            out = m["predict"](live)
            col = out.iloc[:, 0]
            assert out.index.equals(live.index), "predictions must cover the downloaded ids exactly"
            assert np.isfinite(col).all() and col.between(0, 1).all(), "predictions must be finite in [0, 1]"
            pred_path = DATA_DIR / f"round_{window or 'none'}_{split}_{m['name']}.parquet"
            pd.DataFrame({"id": out.index.astype(str), "prediction": col.to_numpy()}).to_parquet(pred_path, index=False)
            m["pred_path"] = pred_path
            log(f"3 predict {m['name']}: {len(out)} rows, range [{col.min():.4f}, {col.max():.4f}] -> {pred_path}")
        except Exception as e:
            log_error(f"predict {m['name']}", e)

    # Re-read the fence right before spending uploads: the round may have closed while we predicted.
    if not dry:
        try:
            cad_now = client.get_status().get("cadence") or {}
            if cad_now.get("intake_fenced") is not False or cad_now.get("open_window") != window:
                log(f"ABORT: round changed before submit (now {cad_now.get('open_window')}, "
                    f"fenced={cad_now.get('intake_fenced')}). Re-run for the new round.")
                raise SystemExit(1)
        except SystemExit:
            raise
        except Exception as e:
            log_error("pre-submit get_status", e)
            log("   continuing: the round lane refuses a fenced upload server-side")

    landed = {}
    if stake_m:
        # 4. stake model first
        log(f"4 submit stake model {stake_m['name']}")
        landed[stake_m["name"]] = submit(client, stake_m, window, dry)

        # 5. stake draft, right after the stake model's upload succeeds
        if landed[stake_m["name"]]:
            log(f"5 stake draft on {stake_m['name']}")
            try:
                draft_stake(client, stake_m, args.amount, window, dry)
            except Exception as e:
                log_error("stake draft", e)
        else:
            log(f"5 no draft: {stake_m['name']}'s submission did not land")
    else:
        log("4-5 skipped: no --stake-model")

    # 6. the rest
    for i, m in enumerate(models[1 if stake_m else 0:], start=1):
        log(f"6 submit [{i}] {m['name']}")
        landed[m["name"]] = submit(client, m, window, dry)

    log(f"done: all_succeeded={all(landed.values())} "
        + " ".join(f"{n}={'OK' if ok else 'FAILED'}" for n, ok in landed.items()))


if __name__ == "__main__":
    main()
