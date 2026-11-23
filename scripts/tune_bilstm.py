"""CLI HPO BiLSTM (Word2Vec + BiLSTM).

Contoh:
    python scripts/tune_bilstm.py \
        --csv-input ../old/ocr_output/data-mengandung.csv \
        --preset standard --trials 30 --strategy random \
        --output-dir artifacts/hpo/bilstm_standard --retrain-best
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click

from app.experiment.artifacts import create_run_dir, write_json
from app.experiment.bilstm_pipeline import (
    build_vectorizers,
    run_bilstm_trial,
    train_final_bilstm,
    vectorizer_key,
)
from app.experiment.configs import BiLSTMConfig, DataConfig, HPOConfig
from app.experiment.data_pipeline import prepare_data
from app.experiment.gpu import distributed_world_size
from app.experiment.hpo import HPOEngine
from app.experiment.search_spaces import bilstm_search_space, preset_trials


@click.command()
@click.option("--csv-input", required=True, help="CSV sumber (delimiter ;).")
@click.option("--frozen-holdout", default=None, help="JSON daftar frozen holdout.")
@click.option("--preset", default="standard", type=click.Choice(["smoke", "standard", "full"]))
@click.option("--trials", default=None, type=int, help="Jumlah trial (default mengikuti preset).")
@click.option("--strategy", default="random", type=click.Choice(["grid", "random", "bayesian"]))
@click.option("--seed", default=42, type=int)
@click.option("--objective", default="val_f1", help="Metrik seleksi (val_f1/val_roc_auc/val_loss).")
@click.option("--timeout-minutes", default=None, type=float)
@click.option("--output-dir", default="./artifacts/hpo/bilstm")
@click.option("--cross-device-ops", type=click.Choice(["auto", "nccl", "hierarchical", "reduction_to_one"]), default=None)
@click.option("--retrain-best/--no-retrain-best", default=False)
def main(
    csv_input: str,
    frozen_holdout: str | None,
    preset: str,
    trials: int | None,
    strategy: str,
    seed: int,
    objective: str,
    timeout_minutes: float | None,
    output_dir: str,
    cross_device_ops: str | None,
    retrain_best: bool,
) -> None:
    if distributed_world_size() > 1:
        raise click.UsageError(
            "HPO BiLSTM harus single-process; MirroredStrategy memakai semua GPU "
            "di dalam satu proses. Jalankan satu proses saja."
        )
    from app.repro import apply_reproducibility_lock, seed_all

    apply_reproducibility_lock(seed)
    seed_all(seed)

    data_config = DataConfig(
        csv_input=csv_input,
        frozen_holdout_path=frozen_holdout,
        sort_source=True,
        output_dir="./artifacts/data",
        bert_input_dir="./artifacts/bert/input",
    )
    bundle = prepare_data(data_config)
    space = bilstm_search_space(preset)
    n_trials = trials or preset_trials(preset)

    run_dir = create_run_dir(output_dir, f"bilstm_{preset}", seed)
    hpo_config = HPOConfig(
        strategy=strategy,
        n_trials=n_trials,
        seed=seed,
        timeout_minutes=timeout_minutes,
        objective_metric=objective,
        objective_mode="min" if objective.endswith("loss") else "max",
        output_dir=output_dir,
        study_name=f"bilstm_{preset}",
        resume=True,
    )

    cache: dict[str, object] = {}

    def objective_fn(params: dict, trial_index: int, trial_dir) -> dict:
        config = BiLSTMConfig.from_dict(params)
        if cross_device_ops is not None:
            config.cross_device_ops = cross_device_ops
        key = vectorizer_key(config)
        vectorized = cache.get(key)
        if vectorized is None:
            if len(cache) >= 2:
                cache.clear()
            vectorized = build_vectorizers(bundle, config)
            cache[key] = vectorized
        result = run_bilstm_trial(
            bundle, config, trial_dir,
            vectorized=vectorized,
            save_model=False, save_predictions=False,
            evaluate_holdout=False, verbose=0,
        )
        return {"metrics": result.metrics, "artifacts": result.artifacts}

    engine = HPOEngine(hpo_config, space, run_dir=run_dir, verbose=True)
    result = engine.run(objective_fn)
    click.echo(f"Run dir: {result.run_dir}")
    if result.best is None:
        raise SystemExit("Tidak ada trial sukses.")
    click.echo(f"Best trial: {result.best.trial_id}")
    click.echo(f"Best objective ({objective}): {result.best.metrics.get('objective')}")
    write_json(result.run_dir / "best_params.json", result.best.to_record())

    if retrain_best:
        best_config = BiLSTMConfig.from_dict(result.best.params)
        if cross_device_ops is not None:
            best_config.cross_device_ops = cross_device_ops
        final_dir = os.path.join(output_dir, "best_model")
        output = train_final_bilstm(
            bundle,
            best_config,
            output_dir=os.path.join(final_dir, "output"),
            model_dir=os.path.join(final_dir, "models"),
            hpo_info={
                "source": str(result.run_dir),
                "best_trial": result.best.trial_id,
                "best_params": result.best.params,
                "best_metrics": result.best.metrics,
                "search_space": space.to_dict(),
                "n_trials": len(result.trials),
            },
            verbose=1,
        )
        click.echo(output["eval_table"].to_string(index=False))
        click.echo(f"Model final: {output['model_dir']}")


if __name__ == "__main__":
    main()
