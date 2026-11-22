"""Paket eksperimen: konfigurasi, data, HPO, training, dan pelaporan.

Impor modul berat (TensorFlow/Torch) dilakukan secara lazy di dalam fungsi,
sehingga paket ini tetap bisa diimpor di environment ringan.
"""

from app.experiment.configs import (
    BertConfig,
    BiLSTMConfig,
    ConfigError,
    DataConfig,
    HPOConfig,
)
from app.experiment.data_pipeline import DataBundle, prepare_data, write_bert_inputs
from app.experiment.hpo import HPOEngine, HPOResult, TrialOutcome, run_hpo
from app.experiment.search_spaces import (
    SearchSpace,
    bert_search_space,
    bilstm_search_space,
    preset_trials,
    space_summary,
)

__all__ = [
    "BertConfig",
    "BiLSTMConfig",
    "ConfigError",
    "DataConfig",
    "DataBundle",
    "HPOConfig",
    "HPOEngine",
    "HPOResult",
    "TrialOutcome",
    "SearchSpace",
    "bert_search_space",
    "bilstm_search_space",
    "prepare_data",
    "preset_trials",
    "run_hpo",
    "space_summary",
    "write_bert_inputs",
]
