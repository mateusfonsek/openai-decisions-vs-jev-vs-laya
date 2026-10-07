from pathlib import Path

from bench.cli import main

ROOT = Path(__file__).parent.parent


def test_validate_runs_on_repo_dataset_dir(tmp_path, capsys):
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "suites.yaml").write_text((ROOT / "dataset" / "suites.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    for s in ["routing", "injection", "judge"]:
        (d / f"{s}.jsonl").write_text("", encoding="utf-8")
    code = main(["validate", "--dataset", str(d), "--no-balance"])
    assert code == 0


def test_run_without_key_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("bench.cli.load_dotenv", lambda *a, **k: None)
    code = main(["run", "--provider", "openai", "--results", str(tmp_path)])
    assert code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
