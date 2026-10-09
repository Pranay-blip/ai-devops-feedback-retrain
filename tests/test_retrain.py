import src.retrain as retrain


def test_retrain_skips_when_not_enough_feedback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(retrain, "FEEDBACK", tmp_path / "fb.csv")
    monkeypatch.setattr("sys.argv", ["retrain"])

    assert retrain.main() == 0
    assert "SKIPPED" in capsys.readouterr().out