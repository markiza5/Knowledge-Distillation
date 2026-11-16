import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "src" / "teacher_analysis.py"
analysis = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location("teacher_analysis", MODULE)
    analysis = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analysis
    spec.loader.exec_module(analysis)


class TeacherAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(analysis, "Analysis module has not been implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        truth = [0] * 8 + list(range(1, 8))
        predicted = [0] * 9 + list(range(2, 8))
        probabilities = np.eye(8)[predicted]
        frame = pd.DataFrame({"index": range(15), "label": truth, "prediction": predicted})
        for i in range(8):
            frame[f"p_{i}"] = probabilities[:, i]
        matrix = np.diag([8, 0, 1, 1, 1, 1, 1, 1])
        matrix[1, 0] = 1
        metrics = {"accuracy": 14 / 15, "balanced_accuracy": 7 / 8, "n_samples": 15, "confusion_matrix": matrix.tolist(), "recall_per_class": [1, 0, 1, 1, 1, 1, 1, 1]}
        self.report = {"primary_teacher_seed": 71397589, "data": {"class_names": {str(i): f"Class {i}" for i in range(8)}}, "teachers": [{"seed": 71397589, "source_test_balanced_accuracy": 7 / 8, "splits": {"val": metrics, "test": metrics}}]}
        (self.results / "teacher_reference.json").write_text(json.dumps(self.report), encoding="utf-8")
        for split in ("val", "test"):
            frame.to_csv(self.results / f"teacher_71397589_{split}_predictions.csv", index=False)

    def test_recomputes_imbalanced_metrics_and_primary_role(self):
        bundle = analysis.load_teacher_analysis(self.results)
        table = analysis.teacher_table(bundle)
        self.assertAlmostEqual(table.iloc[0]["Test accuracy (%)"], 100 * 14 / 15)
        self.assertAlmostEqual(table.iloc[0]["Test balanced accuracy (%)"], 87.5)
        self.assertEqual(table.iloc[0]["Role"], "Fixed KD teacher")
        self.assertEqual(table.iloc[0]["Source difference (pp)"], 0)
        classes = analysis.per_class_table(bundle, 71397589, "test")
        self.assertEqual(classes.iloc[0]["Support"], 8)
        self.assertEqual(classes.iloc[1]["Recall (%)"], 0)

    def test_corrupt_probabilities_or_indices_are_rejected(self):
        path = self.results / "teacher_71397589_test_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "p_0"] = 0.5
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "probabilit"):
            analysis.load_teacher_analysis(self.results)
        frame.loc[0, "p_0"] = 1
        frame.loc[1, "index"] = 0
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "index"):
            analysis.load_teacher_analysis(self.results)

    def test_summary_disagreement_is_rejected(self):
        self.report["teachers"][0]["splits"]["test"]["accuracy"] = 0.1
        (self.results / "teacher_reference.json").write_text(json.dumps(self.report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "accuracy"):
            analysis.load_teacher_analysis(self.results)

    def test_fixed_teacher_cannot_be_reselected_from_reference_results(self):
        second = json.loads(json.dumps(self.report["teachers"][0]))
        second["seed"] = 133560673
        self.report["teachers"].append(second)
        self.report["primary_teacher_seed"] = 133560673
        for split in ("val", "test"):
            content = (self.results / f"teacher_71397589_{split}_predictions.csv").read_bytes()
            (self.results / f"teacher_133560673_{split}_predictions.csv").write_bytes(content)
        (self.results / "teacher_reference.json").write_text(json.dumps(self.report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fixed"):
            analysis.load_teacher_analysis(self.results)

    def test_invalid_class_map_or_missing_results_fail_without_download(self):
        del self.report["data"]["class_names"]["7"]
        (self.results / "teacher_reference.json").write_text(json.dumps(self.report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "class"):
            analysis.load_teacher_analysis(self.results)
        with self.assertRaises(FileNotFoundError):
            analysis.load_teacher_analysis(self.root / "missing")

    def test_root_discovery_works_from_notebooks_and_rejects_unrelated_folder(self):
        notebooks = self.root / "notebooks"
        notebooks.mkdir()
        self.assertEqual(analysis.find_project_root(notebooks), self.root)
        with tempfile.TemporaryDirectory() as other:
            with self.assertRaises(FileNotFoundError):
                analysis.find_project_root(Path(other))


if __name__ == "__main__":
    unittest.main()
