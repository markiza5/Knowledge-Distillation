# Knowledge Distillation for Blood Cell Classification

An academic Deep Learning project investigating whether output-based Knowledge Distillation improves a small CNN compared with conventional training of the same architecture on BloodMNIST.

- **Data:** eight-class BloodMNIST, RGB images at 224 × 224 pixels, official train/validation/test splits.
- **Fixed teacher:** pretrained DenseNet-121 checkpoint with seed `71397589`.
- **Planned student:** four convolutional blocks, global average pooling, and an eight-class linear head.
- **Primary metric:** Balanced Accuracy; supplementary Accuracy, per-class analysis, model size, and inference latency.
- **Current status:** environment and pretrained teacher verified. Student training and KD experiments have not been implemented or run.

## One central notebook

Open [KD project notebook](notebooks/KD%20project%20notebook.ipynb). It contains the initial data exploration and the teacher-analysis chapter in English. Future completed baseline/KD experiments will be documented in the same notebook. Shared implementation code stays in `src/`.

| Mode | What runs | Required environment and artifacts |
|---|---|---|
| `analysis` — default | Saved data profile, sample preview, recomputed teacher metrics, charts and confusion matrices | Analysis requirements and versioned result files; no raw dataset, weights or GPU |
| `full` | Additionally loads the local dataset and repeats the initial exploration | Full requirements and local raw data; optional teacher reevaluation also needs weights and CUDA |

The first code cell controls execution:

```python
RUN_MODE = "analysis"
ALLOW_DOWNLOADS = False
REEVALUATE_TEACHER = False
```

Select **Restart Kernel and Run All** after changing modes. Analysis rejects download/reevaluation flags. Full mode still does not download weights or reevaluate teachers unless explicitly requested. There is no student training code in either route.

### What the notebook shows

1. Data split counts, class names, original first-eight-training-image preview, and training class distribution.
2. Verification protocol and the single fixed KD teacher.
3. Validation/test Accuracy and Balanced Accuracy, recomputed from saved CSV predictions.
4. Source comparison and reference variability across three published weight checkpoints.
5. Fixed-teacher test confusion matrices, counts and row-normalized percentages.
6. Per-class recall, support, error pairs, interpretation and limitations.
7. Explicitly gated optional teacher reevaluation and placeholders for future student/KD results.

The two additional published checkpoints are reference runs of the same architecture. They are not an ensemble, and the best test score is not used to choose the teacher.

## Requirements files

| File | Purpose |
|---|---|
| `requirements-analysis.txt` | Notebook analysis, tables and charts without model libraries |
| `requirements.txt` | Full environment; includes the analysis requirements and adds data/model libraries |
| `requirements-lock-gpu.txt` | Snapshot of installed versions from the verified Windows GPU environment, including indirect dependencies |

The GPU snapshot includes `timm==1.0.31` and `safetensors==0.8.0`. They passed import, GPU DenseNet forward and tensor save/load checks without changing previously installed package versions. The pretrained checkpoints were subsequently verified separately. Reinstallation of the entire GPU snapshot into a fresh environment has not been tested.

## Partner setup: analysis only

Use Python 3.12 where possible; the development Python is 3.12.7. The following commands are for PowerShell on Windows, from the project root after clone/pull. The partner's actual operating system has not been inspected; adjust interpreter paths on other platforms.

For a **new** environment only:

```powershell
py -3.12 -m venv .venv
```

If `.venv` already exists, use it rather than recreating it. Then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m jupyter lab
```

Select this environment's Python as the notebook kernel. Keep `RUN_MODE="analysis"` and both opt-in flags `False`. No PyTorch, CUDA, raw-image download or model-weight download is needed for this route. The installation commands install Python packages; notebook analysis itself does not access the network.

On Windows, keep the environment path reasonably short. During verification, installing JupyterLab into a deeply nested temporary project path encountered the Windows path-length limit; a fresh environment at a shorter temporary path was used instead, without changing Windows settings.

Required analysis files:

- `notebooks/KD project notebook.ipynb`
- `src/teacher_analysis.py`
- `results/teacher_reference.json`
- All six `results/teacher_<seed>_<split>_predictions.csv` files
- `results/data_summary.json`
- `results/figures/data_samples.png`

The notebook raises a clear error if inputs are missing or inconsistent; it never silently runs inference to regenerate them. The saved preview contains only eight illustrative training images, not the full dataset.

## Full setup on the NVIDIA training machine

Verified hardware: RTX 4050 Laptop with 6GB VRAM. The existing environment has PyTorch `2.13.0+cu130` and torchvision `0.28.0+cu130`. Do not reinstall these if the verified environment is already working.

For a new compatible Windows/NVIDIA environment, first install the CUDA build:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Check driver/platform compatibility on another machine rather than assuming this command works everywhere. `requirements.txt` pins public PyTorch versions without a CUDA suffix; installing the appropriate CPU/CUDA build first determines the compute platform.

Set `RUN_MODE="full"` to load existing `data/bloodmnist_224.npz` and repeat the original exploration. Downloads remain disabled unless `ALLOW_DOWNLOADS=True` is explicitly set. To intentionally reevaluate teachers, also set `REEVALUATE_TEACHER=True`; local checkpoints and CUDA must be available. Reevaluation writes a new timestamped results directory and does not overwrite the saved analysis results. No student training is triggered.

### Optional full CPU environment

For local raw-data exploration or model-code inspection without CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Use a separate new environment when changing compute builds. Raw-data exploration can run on CPU; the existing teacher-verification script intentionally requires CUDA, so keep teacher reevaluation disabled in this route. Full training is planned only on the GPU machine.

## Saved results and provenance

The fixed teacher achieved **98.2223% test Balanced Accuracy** and **98.4215% test Accuracy**, matching the explicit source Balanced Accuracy. Three published checkpoints reproduce the reported reference mean **98.4756%** and sample SD **0.2251 percentage points**. These are inference checks of previously trained weights, not locally repeated teacher training.

- [Teacher verification report](docs/teacher_verification.md)
- `results/teacher_reference.json`: measured metrics, class mapping, environment and checkpoint manifests.
- `results/teacher_source_reference.json`: pinned upstream code references.
- `configs/teachers.json`: pinned repositories, revisions and file checksums.
- `src/verify_teacher.py`: offline evaluation using local data/weights and CUDA.
- `tests/test_teacher_verification.py`: full-environment verification tests.
- `tests/test_teacher_analysis.py`: analysis-only validation tests.

Run analysis tests without model dependencies:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_teacher_analysis.py -v
```

Run the entire test suite in the full environment:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

See `docs/notebook_verification.md` for the isolated analysis execution check and its limits. The English notebook is the central narrative; the earlier teacher-verification report remains a historical supporting record.

## Git and reproducibility

Commit code, notebooks, requirements, README, configs, small JSON/CSV results, and the small saved preview. Both partners need the same result files. Do not commit `.venv/`, raw `data/`, or `checkpoints/`; they are excluded by `.gitignore`. Do not copy a virtual environment between machines.

Add direct analysis dependencies to `requirements-analysis.txt`; add model dependencies to `requirements.txt`. Keep the GPU snapshot separate from partner requirements. The notebook addition uses the existing analysis requirements without adding packages.

To restore the current Windows GPU package snapshot, first install the matching CUDA PyTorch build, then install `requirements-lock-gpu.txt` and run `pip check`. A package snapshot alone does not guarantee reproducible research: retain seeds, run settings, data identity, checkpoint revisions and saved predictions as well.

## References

- [pip requirements format](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
- [PyTorch installation by platform](https://pytorch.org/get-started/locally/)
- [Fixed teacher model card](https://huggingface.co/sdoerrich97/colorist_densenet121_bloodmnist_s71397589)
- [MedMNIST data](https://zenodo.org/records/10519652)
