import json

import pytest

from benchmark import freeze, run_eval
from src import config as cfg


def test_current_config_covers_every_setting_that_changes_answers():
    config = freeze.current_config()
    for key in ("chunk_strategy", "chunk_max_words", "retrieval_mode", "top_k", "retrieval_candidates",
                "rrf_k", "embedding_model", "reranker_model", "context_steps", "generation_model",
                "generation_effort", "system_prompt_sha256", "judge_model", "rubric_sha256", "chunks_sha256"):
        assert key in config
    assert config["retrieval_mode"] == cfg.RETRIEVAL_MODE and config["top_k"] == cfg.TOP_K


def test_write_freeze_saves_the_config_with_a_timestamp_and_never_overwrites(tmp_path):
    path = tmp_path / "frozen_config.json"
    freeze.write_freeze(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["config"] == freeze.current_config()
    assert saved["frozen_at"].endswith("+00:00") and "git_commit" in saved
    with pytest.raises(FileExistsError):
        freeze.write_freeze(path)


def test_changed_keys_is_empty_until_a_setting_changes(tmp_path, monkeypatch):
    path = tmp_path / "frozen_config.json"
    freeze.write_freeze(path)
    assert freeze.changed_keys(path) == []
    monkeypatch.setattr(cfg, "TOP_K", cfg.TOP_K + 1)
    monkeypatch.setattr(cfg, "CONTEXT_STEPS", ())
    assert freeze.changed_keys(path) == ["context_steps", "top_k"]


def test_test_split_runs_only_while_the_config_still_matches_the_freeze(tmp_path, monkeypatch):
    path = tmp_path / "frozen_config.json"
    freeze.write_freeze(path)
    run_eval.check_split_allowed("test", verify=lambda: True, frozen_path=path)   # no exit
    monkeypatch.setattr(cfg, "RETRIEVAL_MODE", "vector")
    with pytest.raises(SystemExit) as e:
        run_eval.check_split_allowed("test", verify=lambda: True, frozen_path=path)
    assert "retrieval_mode" in str(e.value)
