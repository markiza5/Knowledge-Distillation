"""Validate and analyze saved teacher results without model dependencies."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def find_project_root(start: Path) -> Path:
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "results" / "teacher_reference.json").is_file():
            return candidate
    raise FileNotFoundError("Open the notebook from the project root or notebooks directory; saved results/teacher_reference.json is required.")


def load_teacher_analysis(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    report_path = results_dir / "teacher_reference.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing saved report: {report_path}. Pull the results from Git; analysis never downloads data or weights.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("primary_teacher_seed") != 71397589:
        raise ValueError("The fixed KD teacher must remain seed 71397589; reference test results cannot select a replacement")
    class_map = report["data"]["class_names"]
    if set(class_map) != {str(i) for i in range(8)} or len(set(class_map.values())) != 8:
        raise ValueError("Expected eight distinct class names keyed by labels 0 through 7")
    seeds = [teacher["seed"] for teacher in report["teachers"]]
    if len(set(seeds)) != len(seeds) or report["primary_teacher_seed"] not in seeds:
        raise ValueError("Teacher seeds must be unique and contain the fixed primary teacher")
    bundle = {"report": report, "class_names": [class_map[str(i)] for i in range(8)], "predictions": {}, "metrics": {}}
    for teacher in report["teachers"]:
        seed = teacher["seed"]
        for split in ("val", "test"):
            path = results_dir / f"teacher_{seed}_{split}_predictions.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved predictions: {path}. Analysis does not run inference.")
            frame = pd.read_csv(path)
            columns = ["index", "label", "prediction"] + [f"p_{i}" for i in range(8)]
            if list(frame.columns) != columns or frame.empty:
                raise ValueError(f"Unexpected columns or empty predictions: {path.name}")
            if not np.array_equal(frame["index"].to_numpy(), np.arange(len(frame))):
                raise ValueError(f"Prediction index must be unique and ordered: {path.name}")
            labels = frame[["label", "prediction"]].to_numpy(dtype=float)
            if not np.isfinite(labels).all() or (labels != np.floor(labels)).any() or (labels < 0).any() or (labels > 7).any():
                raise ValueError(f"Invalid class labels: {path.name}")
            truth, predicted = labels.astype(int).T
            if set(np.unique(truth)) != set(range(8)):
                raise ValueError(f"Missing ground-truth class: {path.name}")
            probabilities = frame[[f"p_{i}" for i in range(8)]].to_numpy(dtype=float)
            if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0):
                raise ValueError(f"Invalid probabilities: {path.name}")
            if not np.array_equal(predicted, probabilities.argmax(axis=1)):
                raise ValueError(f"Prediction disagrees with probabilities: {path.name}")
            matrix = confusion_matrix(truth, predicted, labels=np.arange(8))
            computed = {"accuracy": float(accuracy_score(truth, predicted)), "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)), "n_samples": len(frame), "confusion_matrix": matrix, "recall_per_class": matrix.diagonal() / matrix.sum(axis=1)}
            saved = teacher["splits"][split]
            for metric in ("accuracy", "balanced_accuracy", "n_samples"):
                if not np.isclose(computed[metric], saved[metric], atol=1e-12, rtol=0):
                    raise ValueError(f"Saved {metric} disagrees with CSV: {path.name}")
            if not np.array_equal(matrix, saved["confusion_matrix"]) or not np.allclose(computed["recall_per_class"], saved["recall_per_class"], atol=1e-12, rtol=0):
                raise ValueError(f"Saved per-class metrics disagree with CSV: {path.name}")
            bundle["predictions"][(seed, split)] = frame
            bundle["metrics"][(seed, split)] = computed
    return bundle


def teacher_table(bundle: dict) -> pd.DataFrame:
    rows = []
    primary_seed = bundle["report"]["primary_teacher_seed"]
    for teacher in bundle["report"]["teachers"]:
        seed = teacher["seed"]
        val, test = (bundle["metrics"][(seed, split)] for split in ("val", "test"))
        source = teacher["source_test_balanced_accuracy"]
        rows.append({"Seed": str(seed), "Role": "Fixed KD teacher" if seed == primary_seed else "Reference only", "Validation accuracy (%)": 100 * val["accuracy"], "Validation balanced accuracy (%)": 100 * val["balanced_accuracy"], "Test accuracy (%)": 100 * test["accuracy"], "Test balanced accuracy (%)": 100 * test["balanced_accuracy"], "Source test balanced accuracy (%)": 100 * source, "Source difference (pp)": 100 * (test["balanced_accuracy"] - source)})
    return pd.DataFrame(rows)


def per_class_table(bundle: dict, seed: int, split: str) -> pd.DataFrame:
    matrix = bundle["metrics"][(seed, split)]["confusion_matrix"]
    support = matrix.sum(axis=1)
    return pd.DataFrame({"Label": np.arange(8), "Class": bundle["class_names"], "Support": support, "Correct": matrix.diagonal(), "Errors": support - matrix.diagonal(), "Recall (%)": 100 * matrix.diagonal() / support})
