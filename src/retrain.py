import argparse
import json
import shutil
import sys
from datetime import datetime

from src.train import FEEDBACK, MODELS, train_candidate

MIN_FEEDBACK = 20


def feedback_count():
    if not FEEDBACK.exists():
        return 0
    with FEEDBACK.open() as f:
        return max(sum(1 for _ in f) - 1, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain even if feedback is low")
    args = parser.parse_args()

    n = feedback_count()
    if n < MIN_FEEDBACK and not args.force:
        print(f"SKIPPED: only {n}/{MIN_FEEDBACK} feedback samples collected")
        return 0

    # 1. Train candidate (exits with an error if it fails the quality gate)
    candidate_acc = train_candidate()
    cand_meta = json.loads((MODELS / "candidate_meta.json").read_text())

    # 2. Compare with the live model
    live_meta_path = MODELS / "model_meta.json"
    live_meta = json.loads(live_meta_path.read_text()) if live_meta_path.exists() else None
    live_acc = live_meta["accuracy"] if live_meta else 0.0
    live_version = live_meta.get("version", 1) if live_meta else 0

    if round(candidate_acc, 4) < live_acc:
        sys.exit(f"REJECTED: candidate {candidate_acc:.4f} is worse than live {live_acc:.4f}")

    # 3. Archive the old model
    if live_meta:
        archive = MODELS / "archive"
        archive.mkdir(exist_ok=True)
        shutil.copy(MODELS / "model.joblib", archive / f"model_v{live_version}.joblib")
        shutil.copy(live_meta_path, archive / f"model_v{live_version}_meta.json")

    # 4. Promote the candidate
    cand_meta["version"] = live_version + 1
    cand_meta["promoted_at"] = datetime.now().isoformat(timespec="seconds")
    shutil.copy(MODELS / "candidate.joblib", MODELS / "model.joblib")
    live_meta_path.write_text(json.dumps(cand_meta, indent=2))
    print(f"PROMOTED: v{live_version} -> v{live_version + 1} "
          f"(accuracy {live_acc:.4f} -> {candidate_acc:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())