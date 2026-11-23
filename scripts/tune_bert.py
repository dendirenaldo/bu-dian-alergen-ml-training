"""CLI HPO IndoBERT.

Contoh (GPU):
    python scripts/tune_bert.py \
        --csv-input ../old/ocr_output/data-mengandung.csv \
        --preset standard --trials 20 --strategy random \
        --output-dir artifacts/hpo/bert_standard --retrain-best
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click

from app.experiment.artifacts import create_run_dir, write_json
from app.experiment.bert_pipeline import run_bert_trial, train_final_bert
from app.experiment.configs import BertConfig, DataConfig, HPOConfig
from app.experiment.data_pipeline import prepare_data
from app.experiment.gpu import distributed_world_size
from app.experiment.hpo import HPOEngine
from app.experiment.search_spaces import bert_search_space, preset_trials


@click.command()
@click.option("--csv-input", required=True, help="CSV sumber (delimiter ;).")
@click.option("--frozen-holdout", default=None, help="JSON daftar frozen holdout.")
@click.option("--preset", default="standard", type=click.Choice(["smoke", "standard", "full"]))
@click.option("--trials", default=None, type=int, help="Jumlah trial (default mengikuti preset).")
@click.option("--strategy", default="random", type=click.Choice(["grid", "random", "bayesian"]))
@click.option("--seed", default=42, type=int)
@click.option("--objective", default="val_f1", help="Metrik seleksi (val_f1/val_roc_auc/val_loss).")
@click.option("--timeout-minutes", default=None, type=float)
@click.option("--output-dir", default="./artifacts/hpo/bert")
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
    retrain_best: bool,
) -> None:
    if distributed_world_size() > 1:
        raise click.UsageError(
            "HPO BERT berjalan sequential (satu trial satu proses). "
            "Untuk memakai banyak GPU, jalankan beberapa proses tune_bert.py "
            "dengan CUDA_VISIBLE_DEVICES berbeda dan output-dir berbeda."
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
    space = bert_search_space(preset)
    n_trials = trials or preset_trials(preset)

    run_dir = create_run_dir(output_dir, f"bert_{preset}", seed)
    hpo_config = HPOConfig(
        strategy=strategy,
        n_trials=n_trials,
        seed=seed,
        timeout_minutes=timeout_minutes,
        objective_metric=objective,
        objective_mode="min" if objective.endswith("loss") else "max",
        output_dir=output_dir,
        study_name=f"bert_{preset}",
        resume=True,
    )

    def objective_fn(params: dict, trial_index: int, trial_dir) -> dict:
        config = BertConfig.from_dict(params)
        result = run_bert_trial(
            bundle, config, trial_dir,
            save_model=False, save_predictions=False,
            evaluate_holdout=False, verbose=False,
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
        best_config = BertConfig.from_dict(result.best.params)
        final_dir = os.path.join(output_dir, "best_model")
        output = train_final_bert(
            bundle,
            best_config,
            output_dir=os.path.join(final_dir, "output"),
            hpo_info={
                "source": str(result.run_dir),
                "best_trial": result.best.trial_id,
                "best_params": result.best.params,
                "best_metrics": result.best.metrics,
                "search_space": space.to_dict(),
                "n_trials": len(result.trials),
            },
            verbose=True,
        )
        click.echo(f"Model final: {output['output_dir']}")


if __name__ == "__main__":
    main()
