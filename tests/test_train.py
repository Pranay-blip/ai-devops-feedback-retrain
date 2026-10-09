from src.train import THRESHOLD, train_candidate


def test_training_meets_quality_gate(tmp_path, monkeypatch):
    # use a temp folder so the real models/ and data/ are never touched
    monkeypatch.setattr("src.train.MODELS", tmp_path)
    monkeypatch.setattr("src.train.FEEDBACK", tmp_path / "no_feedback.csv")

    acc = train_candidate()

    assert acc >= THRESHOLD
    assert (tmp_path / "candidate.joblib").exists()
    assert (tmp_path / "candidate_meta.json").exists()