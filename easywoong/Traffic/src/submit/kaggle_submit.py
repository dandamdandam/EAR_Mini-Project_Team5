"""Upload one validated payload and persist its Kaggle reference and score.

--status-only is read-only. An upload intent prevents accidental duplicate
submissions if the connection is interrupted after Kaggle receives the file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.preprocess.run import sha256

COMPETITION = "2026-ieee-big-data-traffic-flow-bench"


def record(item):
    return {"ref": item.ref, "date": item.date.isoformat() if item.date else None,
            "description": item.description, "file_name": item.file_name,
            "status": getattr(item.status, "name", str(item.status)),
            "public_score": item.public_score or None, "private_score": item.private_score or None,
            "error_description": item.error_description or None}


def run(out, status_only=False):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    metadata = json.loads((out / "metadata.json").read_text())
    intent_path, receipt_path = out / "upload_intent.json", out / "upload_receipt.json"
    if not status_only:
        if intent_path.exists():
            raise FileExistsError("Upload was already attempted. Use --status-only to avoid a duplicate submission.")
        file = out / "submission.csv"
        if sha256(file) != metadata["submission_sha256"]:
            raise ValueError("Payload differs from the validated submission hash")
        before = [record(item) for item in api.competition_submissions(COMPETITION, page_size=100) if item is not None]
        (out / "kaggle_history_before.json").write_text(json.dumps(before, indent=2) + "\n")
        label = metadata.get("submission_description", "STAEformer 5ep + CatBoost queue + NNLS .05")
        description = f"{out.name}: {label} | {metadata['submission_sha256'][:12]}"
        intent = {"competition": COMPETITION, "description": description,
                  "submission_sha256": metadata["submission_sha256"],
                  "started_utc": datetime.now(timezone.utc).isoformat()}
        with intent_path.open("x") as handle:
            json.dump(intent, handle, indent=2)
        response = api.competition_submit(str(file.resolve()), description, COMPETITION, quiet=True)
        receipt = {"ref": response.ref, "message": response.message,
                   "received_utc": datetime.now(timezone.utc).isoformat()}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        if not response.ref:
            raise RuntimeError("Kaggle returned no submission reference; inspect the receipt before retrying")
    intent = json.loads(intent_path.read_text())
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    submissions = [record(item) for item in api.competition_submissions(COMPETITION, page_size=100) if item is not None]
    matches = [item for item in submissions if item["ref"] == receipt.get("ref") or item["description"] == intent["description"]]
    if not matches:
        pending = {"ref": receipt.get("ref"), "status": "NOT_YET_LISTED",
                   "competition": COMPETITION, "description": intent["description"],
                   "submission_sha256": metadata["submission_sha256"],
                   "checked_utc": datetime.now(timezone.utc).isoformat()}
        (out / "result.json").write_text(json.dumps(pending, indent=2) + "\n")
        print(json.dumps(pending, indent=2), flush=True)
        return pending
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Kaggle record for this upload; found {len(matches)}. No new upload was made.")
    result = matches[0]
    previous = json.loads((out / "kaggle_history_before.json").read_text())
    previous_scores = [float(item["public_score"]) for item in previous if item["public_score"]]
    result.update({"competition": COMPETITION, "submission_sha256": metadata["submission_sha256"],
                   "checked_utc": datetime.now(timezone.utc).isoformat(),
                   "previous_best_public_score": max(previous_scores) if previous_scores else None})
    if result["public_score"] and previous_scores:
        result["delta_from_previous_best"] = float(result["public_score"]) - max(previous_scores)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("submissions/003_multitask"))
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    run(args.out, args.status_only)


if __name__ == "__main__":
    main()
