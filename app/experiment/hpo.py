"""Mesin pencarian hiperparameter (HPO) yang ringan dan bisa di-resume.

Fitur:

- Strategi ``grid``, ``random``, dan ``bayesian`` (Optuna TPE bila terpasang).
- Setiap trial dicatat ke ``trials.jsonl`` + ``leaderboard.csv``.
- Resume otomatis: trial dengan parameter sama yang sudah sukses dilewati.
- Timeout, penanganan trial gagal, dan penyimpanan konfigurasi terbaik.
- Tidak mengimpor TensorFlow/Torch; runner model dioper sebagai callable.

Contoh pemakaian di notebook::

    engine = HPOEngine(hpo_config, search_space)
    result = engine.run(lambda params, i, trial_dir: run_trial(params, i, trial_dir))
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.experiment.artifacts import (
    append_jsonl,
    create_run_dir,
    environment_snapshot,
    params_hash,
    read_jsonl,
    to_jsonable,
    utc_timestamp,
    write_csv,
    write_json,
)
from app.experiment.configs import ConfigError, HPOConfig
from app.experiment.search_spaces import SearchSpace

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[dict[str, Any], int, Path], dict[str, Any]]


@dataclass
class TrialOutcome:
    """Hasil satu trial HPO."""

    trial_id: str
    params: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "ok"  # ok | failed | timeout
    duration_sec: float = 0.0
    error: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    params_hash: str = ""
    seed: int | None = None

    @property
    def objective_value(self) -> float | None:
        value = self.metrics.get("objective")
        return float(value) if value is not None else None

    def to_record(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "trial_id": self.trial_id,
                "status": self.status,
                "params": self.params,
                "params_hash": self.params_hash,
                "seed": self.seed,
                "metrics": self.metrics,
                "duration_sec": self.duration_sec,
                "error": self.error,
                "artifacts": self.artifacts,
            }
        )


@dataclass
class HPOResult:
    """Ringkasan satu sesi HPO."""

    run_dir: Path
    trials: list[TrialOutcome]
    best: TrialOutcome | None
    objective_metric: str
    objective_mode: str
    summary: dict[str, Any]


def _extract_metrics(result: Any) -> tuple[dict[str, float], dict[str, str]]:
    """Normalisasi return value objective function."""
    if not isinstance(result, dict):
        raise TypeError(
            "Objective function harus mengembalikan dict, "
            "mis. {'metrics': {...}, 'artifacts': {...}}."
        )
    if "metrics" in result:
        metrics = dict(result.get("metrics") or {})
        artifacts = dict(result.get("artifacts") or {})
    else:
        artifacts = dict(result.get("artifacts") or {})
        metrics = {k: v for k, v in result.items() if k != "artifacts"}
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            clean[str(key)] = float(value)
    return clean, {str(k): str(v) for k, v in artifacts.items()}


class HPOEngine:
    """Orkestrator HPO generik."""

    def __init__(
        self,
        config: HPOConfig,
        search_space: SearchSpace,
        run_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.search_space = search_space
        self.verbose = verbose
        if run_dir is None:
            run_dir = create_run_dir(config.output_dir, config.study_name, config.seed)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.run_dir / "trials.jsonl"
        self.leaderboard_path = self.run_dir / "leaderboard.csv"
        self.logger = logger

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_setup(self) -> None:
        write_json(self.run_dir / "hpo_config.json", self.config.to_dict())
        write_json(self.run_dir / "search_space.json", self.search_space.to_dict())
        write_json(self.run_dir / "environment.json", environment_snapshot(seed=self.config.seed))
        if not (self.run_dir / "run_info.json").exists():
            write_json(
                self.run_dir / "run_info.json",
                {
                    "study_name": self.config.study_name,
                    "created_utc": utc_timestamp(),
                    "strategy": self.config.strategy,
                    "objective_metric": self.config.objective_metric,
                    "objective_mode": self.config.objective_mode,
                    "seed": self.config.seed,
                },
            )

    def _load_existing(self) -> list[TrialOutcome]:
        if not self.config.resume:
            return []
        outcomes: list[TrialOutcome] = []
        for record in read_jsonl(self.trials_path):
            try:
                outcomes.append(
                    TrialOutcome(
                        trial_id=str(record.get("trial_id")),
                        params=dict(record.get("params") or {}),
                        metrics={
                            str(k): float(v) for k, v in (record.get("metrics") or {}).items()
                            if isinstance(v, (int, float))
                        },
                        status=str(record.get("status", "ok")),
                        duration_sec=float(record.get("duration_sec") or 0.0),
                        error=record.get("error"),
                        artifacts=dict(record.get("artifacts") or {}),
                        params_hash=str(record.get("params_hash") or ""),
                        seed=record.get("seed"),
                    )
                )
            except Exception:
                self.logger.warning("Baris trials.jsonl rusak dilewati.")
        return outcomes

    def _rewrite_leaderboard(self, outcomes: list[TrialOutcome]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for outcome in outcomes:
            rows.append(
                {
                    "trial_id": outcome.trial_id,
                    "status": outcome.status,
                    "objective": outcome.metrics.get("objective"),
                    "duration_sec": round(outcome.duration_sec, 3),
                    "seed": outcome.seed,
                    "params_json": json.dumps(to_jsonable(outcome.params), ensure_ascii=False),
                    "error": outcome.error or "",
                    **{f"metric_{k}": v for k, v in outcome.metrics.items() if k != "objective"},
                }
            )
        # Urutkan yang sukses di atas, lalu objective terbaik.
        reverse = self.config.objective_mode == "max"
        rows.sort(
            key=lambda row: (
                row["status"] != "ok",
                -(row["objective"] if row["objective"] is not None else float("-inf"))
                if reverse
                else (row["objective"] if row["objective"] is not None else float("inf")),
            )
        )
        metric_keys = sorted(
            {key for outcome in outcomes for key in outcome.metrics if key != "objective"}
        )
        columns = [
            "trial_id", "status", "objective", "duration_sec", "seed",
            "params_json", "error",
        ] + [f"metric_{key}" for key in metric_keys]
        write_csv(self.leaderboard_path, rows, columns=columns)
        return rows

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def _grid_candidates(self, n_candidates: int) -> list[dict[str, Any]]:
        grid = self.search_space.grid(self.config.max_grid_combinations)
        if len(grid) > n_candidates:
            # Ambil sampel deterministik yang tersebar, bukan hanya awal grid.
            step = len(grid) / n_candidates
            grid = [grid[min(len(grid) - 1, int(i * step))] for i in range(n_candidates)]
        return grid

    def _random_candidates(
        self, existing_hashes: set[str], n_candidates: int
    ) -> list[dict[str, Any]]:
        rng = random.Random(self.config.seed)
        candidates: list[dict[str, Any]] = []
        seen = set(existing_hashes)
        attempts = 0
        max_attempts = max(100, n_candidates * 50)
        while len(candidates) < n_candidates and attempts < max_attempts:
            attempts += 1
            params = self.search_space.sample(rng)
            self.search_space.validate(params)
            digest = params_hash(params)
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append(params)
        if len(candidates) < n_candidates:
            self.logger.warning(
                "Hanya mendapat %d/%d kombinasi unik; search space mungkin terlalu kecil.",
                len(candidates), n_candidates,
            )
        return candidates

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, objective_fn: ObjectiveFn) -> HPOResult:
        """Jalankan HPO.

        Args:
            objective_fn: callable ``(params, trial_index, trial_dir) -> dict``.
                ``params`` sudah termasuk ``seed`` per trial.
        """
        self._save_setup()
        existing = self._load_existing()
        outcomes: list[TrialOutcome] = list(existing)
        done_hashes = {o.params_hash for o in existing if o.status == "ok"}
        self.logger.info(
            "HPO mulai: %s | strategi=%s | target=%d trial | resume=%s",
            self.config.study_name, self.config.strategy,
            self.config.n_trials, self.config.resume,
        )

        started = time.monotonic()
        deadline = (
            started + self.config.timeout_minutes * 60
            if self.config.timeout_minutes
            else None
        )

        # ``n_trials`` adalah target TOTAL trial sukses, bukan tambahan.
        # Dengan begitu resume tidak mengulang trial yang sudah selesai.
        n_ok_existing = sum(1 for outcome in existing if outcome.status == "ok")
        remaining = max(0, self.config.n_trials - n_ok_existing)
        if remaining == 0:
            self.logger.info(
                "Target %d trial sudah tercapai (%d sukses); tidak ada trial baru.",
                self.config.n_trials, n_ok_existing,
            )
        if self.config.strategy == "bayesian":
            outcomes = self._run_bayesian(objective_fn, outcomes, deadline, remaining)
        else:
            if self.config.strategy == "grid":
                try:
                    candidates = self._grid_candidates(remaining) if remaining else []
                except ConfigError as exc:
                    self.logger.warning(
                        "Grid tidak bisa dibuat (%s); fallback ke random search.", exc
                    )
                    candidates = self._random_candidates(done_hashes, remaining) if remaining else []
            else:
                candidates = self._random_candidates(done_hashes, remaining) if remaining else []
            for index, params in enumerate(candidates, start=len(existing)):
                if self._deadline_reached(deadline):
                    self.logger.warning("Timeout HPO tercapai; sisa trial dilewati.")
                    break
                digest = params_hash(params)
                if digest in done_hashes:
                    self.logger.info("Trial parameter sama sudah sukses; dilewati.")
                    continue
                outcome = self._execute(objective_fn, params, index)
                outcomes.append(outcome)
                if outcome.status == "ok":
                    done_hashes.add(digest)
                self._rewrite_leaderboard(outcomes)

        best = self._select_best(outcomes)
        rows = self._rewrite_leaderboard(outcomes)
        summary = {
            "study_name": self.config.study_name,
            "strategy": self.config.strategy,
            "objective_metric": self.config.objective_metric,
            "objective_mode": self.config.objective_mode,
            "n_trials_requested": self.config.n_trials,
            "n_trials_recorded": len(outcomes),
            "n_ok": sum(1 for o in outcomes if o.status == "ok"),
            "n_failed": sum(1 for o in outcomes if o.status == "failed"),
            "n_timeout": sum(1 for o in outcomes if o.status == "timeout"),
            "duration_sec": round(time.monotonic() - started, 3),
            "best_trial": best.trial_id if best else None,
            "best_objective": best.metrics.get("objective") if best else None,
            "best_params": best.params if best else None,
            "best_metrics": best.metrics if best else None,
            "run_dir": str(self.run_dir),
            "leaderboard": rows,
        }
        write_json(self.run_dir / "summary.json", summary)
        if best is not None:
            write_json(self.run_dir / "best_params.json", best.to_record())
        self.logger.info(
            "HPO selesai: %d trial, best=%s, objective=%s",
            len(outcomes), best.trial_id if best else None,
            best.metrics.get("objective") if best else None,
        )
        return HPOResult(
            run_dir=self.run_dir,
            trials=outcomes,
            best=best,
            objective_metric=self.config.objective_metric,
            objective_mode=self.config.objective_mode,
            summary=summary,
        )

    def _deadline_reached(self, deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _execute(
        self,
        objective_fn: ObjectiveFn,
        params: dict[str, Any],
        index: int,
    ) -> TrialOutcome:
        trial_id = f"trial_{index:04d}"
        trial_dir = self.run_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        digest = params_hash(params)
        seed = self.config.seed + index
        params_with_seed = {**params, "seed": seed}
        write_json(trial_dir / "params.json", params_with_seed)
        self.logger.info("Mulai %s: %s", trial_id, json.dumps(to_jsonable(params_with_seed)))
        started = time.monotonic()
        try:
            result = objective_fn(params_with_seed, index, trial_dir)
            metrics, artifacts = _extract_metrics(result)
            if self.config.objective_metric not in metrics:
                raise ConfigError(
                    f"Metrik objektif '{self.config.objective_metric}' tidak ada. "
                    f"Metrik tersedia: {sorted(metrics)}"
                )
            metrics["objective"] = metrics[self.config.objective_metric]
            outcome = TrialOutcome(
                trial_id=trial_id,
                params=params_with_seed,
                metrics=metrics,
                status="ok",
                duration_sec=time.monotonic() - started,
                artifacts=artifacts,
                params_hash=digest,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - trial gagal tidak boleh menghentikan HPO
            self.logger.exception("Trial %s gagal: %s", trial_id, exc)
            outcome = TrialOutcome(
                trial_id=trial_id,
                params=params_with_seed,
                status="failed",
                duration_sec=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
                params_hash=digest,
                seed=seed,
            )
        append_jsonl(self.trials_path, outcome.to_record())
        write_json(trial_dir / "result.json", outcome.to_record())
        self.logger.info(
            "%s selesai status=%s objective=%s durasi=%.1fs",
            trial_id, outcome.status,
            outcome.metrics.get("objective"), outcome.duration_sec,
        )
        return outcome

    def _select_best(self, outcomes: list[TrialOutcome]) -> TrialOutcome | None:
        ok = [o for o in outcomes if o.status == "ok" and o.metrics.get("objective") is not None]
        if not ok:
            return None
        reverse = self.config.objective_mode == "max"
        return sorted(ok, key=lambda o: float(o.metrics["objective"]), reverse=reverse)[0]

    # ------------------------------------------------------------------
    # Bayesian (opsional)
    # ------------------------------------------------------------------
    def _run_bayesian(
        self,
        objective_fn: ObjectiveFn,
        outcomes: list[TrialOutcome],
        deadline: float | None,
        n_new_trials: int,
    ) -> list[TrialOutcome]:
        try:
            import optuna
        except ImportError:
            self.logger.warning(
                "Optuna tidak terpasang; strategi bayesian jatuh ke random. "
                "Install dengan: pip install optuna"
            )
            candidates = self._random_candidates(
                {o.params_hash for o in outcomes}, n_new_trials
            )
            for index, params in enumerate(candidates, start=len(outcomes)):
                if self._deadline_reached(deadline):
                    break
                outcomes.append(self._execute(objective_fn, params, index))
                self._rewrite_leaderboard(outcomes)
            return outcomes

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        direction = "maximize" if self.config.objective_mode == "max" else "minimize"
        storage = f"sqlite:///{self.run_dir / 'optuna.db'}"
        try:
            study = optuna.create_study(
                study_name=self.config.study_name,
                direction=direction,
                storage=storage,
                load_if_exists=self.config.resume,
            )
        except Exception:
            study = optuna.create_study(
                study_name=f"{self.config.study_name}_{utc_timestamp()}",
                direction=direction,
                storage=storage,
            )
        self.logger.info(
            "Optuna study dibuat: %s (%d trial selesai)",
            self.config.study_name, len(study.trials),
        )
        start_index = len(outcomes)
        for index in range(start_index, start_index + n_new_trials):
            if self._deadline_reached(deadline):
                self.logger.warning("Timeout HPO tercapai; sisa trial dilewati.")
                break
            trial = study.ask()
            try:
                params = self.search_space.suggest(trial)
                self.search_space.validate(params)
            except Exception as exc:
                self.logger.warning("Saran Optuna tidak valid (%s); trial ditandai gagal.", exc)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue
            outcome = self._execute(objective_fn, params, index)
            outcomes.append(outcome)
            if outcome.status == "ok" and outcome.objective_value is not None:
                study.tell(trial, outcome.objective_value)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            self._rewrite_leaderboard(outcomes)
        try:
            write_json(
                self.run_dir / "optuna_summary.json",
                {
                    "best_value": study.best_value if study.trials else None,
                    "best_params": study.best_params if study.trials else None,
                    "n_trials": len(study.trials),
                },
            )
        except Exception:
            pass
        return outcomes


def run_hpo(
    config: HPOConfig,
    search_space: SearchSpace,
    objective_fn: ObjectiveFn,
    run_dir: str | Path | None = None,
) -> HPOResult:
    """Fungsi pembungkus singkat untuk notebook."""
    engine = HPOEngine(config, search_space, run_dir=run_dir)
    return engine.run(objective_fn)
