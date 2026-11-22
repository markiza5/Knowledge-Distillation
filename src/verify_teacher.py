"""Verify pretrained BloodMNIST teachers by running evaluation."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
from medmnist import INFO, BloodMNIST
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm

SEEDS = (71397589, 133560673, 265017005)


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config):
    expected = {"architecture": "densenet121", "num_classes": 8, "dataset": "bloodmnist", "input_size": 224}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected {key}: {config.get(key)}")
    normalization = config["normalization"]
    mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid RGB normalization")


def load_bloodmnist_split(data_path, split, normalization):
    # MedMNIST derives the filename from the dataset and size.
    if data_path.name != "bloodmnist_224.npz":
        raise ValueError("Expected a file named bloodmnist_224.npz")
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalization["mean"], std=normalization["std"]),
    ])
    return BloodMNIST(
        split=split,
        root=str(data_path.parent),
        size=224,
        as_rgb=True,
        download=False,
        transform=transform,
    )


def load_frozen_weights(model, checkpoint, device):
    state = load_file(str(checkpoint), device="cpu")
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    if len(cleaned) != len(state):
        raise ValueError("Colliding checkpoint keys")
    model.load_state_dict(cleaned, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def load_teacher(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("config.json", "model.safetensors"):
        if file_hash(checkpoint_dir / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    model = timm.create_model(config["architecture"], pretrained=False, num_classes=config["num_classes"])
    return load_frozen_weights(model, checkpoint_dir / "model.safetensors", device), config, manifest


def classification_metrics(truth, predicted):
    truth, predicted = np.asarray(truth).reshape(-1), np.asarray(predicted).reshape(-1)
    if truth.shape != predicted.shape or set(np.unique(truth)) != set(range(8)):
        raise ValueError("Evaluation must contain all eight classes and matching predictions")
    if (predicted < 0).any() or (predicted > 7).any():
        raise ValueError("Invalid predicted class")
    matrix = confusion_matrix(truth, predicted, labels=list(range(8)))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "recall_per_class": (matrix.diagonal() / matrix.sum(axis=1)).tolist(),
        "confusion_matrix": matrix.tolist(),
        "n_samples": len(truth),
    }


def evaluate_classifier(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            logits = model(images.to(device))
            expected_shape = (len(targets), 8)
            if logits.shape != expected_shape:
                raise ValueError("Invalid teacher output")
            has_invalid_scores = not torch.isfinite(logits).all().item()
            if has_invalid_scores:
                raise ValueError("Invalid teacher output")
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
            labels.append(np.asarray(targets).reshape(-1))
            if (batch_index + 1) % 25 == 0:
                print(f"  evaluated batches: {batch_index + 1}/{len(loader)}", flush=True)
    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    return classification_metrics(labels, probabilities.argmax(axis=1)), probabilities


def save_predictions(path, labels, probabilities):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "prediction"] + [f"p_{i}" for i in range(8)])
        for index, (label, values) in enumerate(zip(labels, probabilities)):
            writer.writerow([index, int(label), int(values.argmax())] + values.tolist())


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "bloodmnist_224.npz",
        help="Path to the local bloodmnist_224.npz file",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=root / "checkpoints" / "teachers")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    if (args.output / "teacher_reference.json").exists():
        raise FileExistsError("Choose a new output directory to preserve previous results")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = INFO["bloodmnist"]
    actual_md5 = file_hash(args.data, "md5")
    if actual_md5 != metadata["MD5_224"]:
        raise ValueError("Dataset MD5 does not match official MedMNIST 224 data")
    print("Official BloodMNIST-224 MD5 verified", flush=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_teacher_seed": SEEDS[0],
        "training_performed": False,
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "precision": "float32",
        "data": {
            "filename": args.data.name,
            "md5": actual_md5,
            "sha256": file_hash(args.data),
            "class_names": metadata["label"],
            "source": metadata["url_224"],
        },
        "environment": {
            name: version(name)
            for name in ("torch", "torchvision", "timm", "safetensors", "medmnist", "numpy", "scikit-learn")
        },
        "teachers": [],
    }
    for seed in SEEDS:
        model, config, manifest = load_teacher(args.checkpoint_root / f"seed_{seed}", device)
        if config["seed"] != seed:
            raise ValueError("Teacher seed mismatch")
        record = {
            "seed": seed,
            "manifest": manifest,
            "normalization": config["normalization"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "splits": {},
        }
        state_before = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        print(f"Teacher {seed}: strict weights loaded, frozen, eval mode", flush=True)
        for split in ("val", "test"):
            dataset = load_bloodmnist_split(args.data, split, config["normalization"])
            labels = dataset.labels.reshape(-1)
            if len(labels) != metadata["n_samples"][split]:
                raise ValueError(f"Unexpected size for {split}")
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            start = time.perf_counter()
            metrics, probabilities = evaluate_classifier(model, loader, device)
            metrics["evaluation_seconds_including_loading_batches"] = time.perf_counter() - start
            save_predictions(args.output / f"teacher_{seed}_{split}_predictions.csv", labels, probabilities)
            record["splits"][split] = metrics
            print(f"Teacher {seed} {split}: accuracy={metrics['accuracy']:.8f}, balanced_accuracy={metrics['balanced_accuracy']:.8f}", flush=True)
            del loader, dataset, labels, probabilities
        for name, tensor in model.state_dict().items():
            if not torch.equal(tensor.cpu(), state_before[name]):
                raise RuntimeError(f"Teacher state changed during evaluation: {name}")
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("Teacher is not fully frozen")
        expected = config["metrics"]["test_balanced_accuracy_per_seed"][str(seed)]
        actual = record["splits"]["test"]["balanced_accuracy"]
        record["source_test_balanced_accuracy"] = expected
        record["test_difference_from_source"] = actual - expected
        record["matches_source_to_reported_four_decimals"] = round(actual, 4) == round(expected, 4)
        report["teachers"].append(record)
        (args.output / "teacher_reference.partial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        del model, state_before
        torch.cuda.empty_cache()
    scores = np.array([r["splits"]["test"]["balanced_accuracy"] for r in report["teachers"]])
    report["test_balanced_accuracy_mean"] = float(scores.mean())
    report["test_balanced_accuracy_sample_std"] = float(scores.std(ddof=1))
    report["all_match_source_four_decimals"] = all(r["matches_source_to_reported_four_decimals"] for r in report["teachers"])
    (args.output / "teacher_reference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Evaluation complete; source match:", report["all_match_source_four_decimals"], flush=True)
    print("Test Balanced Accuracy mean/sample std:", scores.mean(), scores.std(ddof=1), flush=True)


if __name__ == "__main__":
    main()
