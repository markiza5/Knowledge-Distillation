import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MODULE = Path(__file__).resolve().parents[1] / "src" / "verify_teacher.py"
verification = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location("verify_teacher", MODULE)
    verification = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = verification
    spec.loader.exec_module(verification)


class TeacherVerificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(verification, "Teacher verification is not implemented")

    def test_preprocessing_scales_rgb_and_uses_config_normalization(self):
        image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        image[..., 1] = 128
        image[..., 2] = 255
        normalization = {"mean": [0.5, 0.25, 0.75], "std": [0.25, 0.5, 1.0]}
        with tempfile.TemporaryDirectory() as temporary:
            data_path = Path(temporary) / "bloodmnist_224.npz"
            np.savez(
                data_path,
                val_images=image,
                val_labels=np.array([[3]]),
                test_images=np.zeros_like(image),
                test_labels=np.array([[5]]),
            )
            dataset = verification.load_bloodmnist_split(data_path, "val", normalization)
            tensor, label = dataset[0]
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(label.item(), 3)
        self.assertTrue(torch.allclose(tensor[:, 0, 0], torch.tensor([-2.0, (128 / 255 - 0.25) / 0.5, 0.25]), atol=1e-6))

    def test_metrics_distinguish_accuracy_from_balanced_accuracy(self):
        truth = np.array([0] * 8 + list(range(1, 8)))
        predicted = np.array([0] * 9 + list(range(2, 8)))
        result = verification.classification_metrics(truth, predicted)
        self.assertAlmostEqual(result["accuracy"], 14 / 15)
        self.assertAlmostEqual(result["balanced_accuracy"], 7 / 8)
        self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (8, 8))
        self.assertEqual(np.asarray(result["confusion_matrix"]).sum(), 15)

    def test_missing_class_is_rejected(self):
        with self.assertRaises(ValueError):
            verification.classification_metrics(np.array([0, 1]), np.array([0, 1]))

    def test_bloodmnist_requires_the_224_filename(self):
        normalization = {"mean": [0, 0, 0], "std": [1, 1, 1]}
        with tempfile.TemporaryDirectory() as temporary:
            data_path = Path(temporary) / "bloodmnist.npz"
            with self.assertRaisesRegex(ValueError, "bloodmnist_224[.]npz"):
                verification.load_bloodmnist_split(data_path, "val", normalization)

    def test_frozen_evaluation_preserves_batchnorm_and_has_no_gradients(self):
        model = nn.Sequential(nn.BatchNorm1d(8), nn.Linear(8, 8, bias=False))
        with torch.no_grad():
            model[1].weight.copy_(torch.eye(8))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.safetensors"
            save_file(model.state_dict(), str(checkpoint))
            verification.load_frozen_weights(model, checkpoint, torch.device("cpu"))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        loader = DataLoader(TensorDataset(torch.eye(8), torch.arange(8)), batch_size=4)
        result, probabilities = verification.evaluate_classifier(model, loader, torch.device("cpu"))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(probabilities.shape, (8, 8))
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)

    def test_wrong_classifier_weights_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.safetensors"
            save_file(nn.Linear(8, 4).state_dict(), str(checkpoint))
            with self.assertRaises(RuntimeError):
                verification.load_frozen_weights(nn.Linear(8, 8), checkpoint, torch.device("cpu"))

    def test_invalid_normalization_is_rejected(self):
        config = {"architecture": "densenet121", "num_classes": 8, "dataset": "bloodmnist", "input_size": 224, "normalization": {"mean": [0.1, 0.2, 0.3], "std": [1.0, 0.0, 1.0]}}
        with self.assertRaises(ValueError):
            verification.validate_config(config)


if __name__ == "__main__":
    unittest.main()
