# Central notebook verification

## Result

Status: **PASS**. The notebook's default analysis route executed from a fresh kernel using only saved analysis artifacts in a separate clean Windows environment. Python version: 3.12.7.

- Executed code cells: 12.
- Embedded chart outputs: 6.
- Analysis unit tests passed in the clean environment: 6.
- Full-environment regression suite: 12 tests passed (teacher verification plus analysis).
- Dependency check: No broken requirements found..
- Fixed teacher test Balanced Accuracy reproduced from CSV: 98.22227711989562%.

## Isolation check

The test fixture contained only the central notebook, the analysis helper and saved JSON/CSV/PNG artifacts. It did not contain data/ or checkpoints/. It ran with torch, torchvision, timm, medmnist and safetensors absent, and CUDA_VISIBLE_DEVICES empty.

Additional Python-level guards rejected model imports, opening raw-data/checkpoint paths, external socket connections and external DNS lookups. Local kernel communication was permitted. All analysis cells completed without triggering these guards. A negative full-mode check failed with the intended missing-local-data error before reaching model-library imports.

The guards are an execution check of the current Python code, not an OS-wide network sandbox. The physical verification machine has an NVIDIA GPU; no GPU API was available to the analysis environment or used by the analysis code. The partner's own computer/OS has not been inspected.

## Dependency setup

Only requirements-analysis.txt and its dependencies were installed in the verification venv. Indirect package versions were constrained to the existing verified package snapshot; a constraints file does not install unrelated model/GPU packages. The actual notebook imports only analysis libraries. The project GPU environment was not changed, and requirements-analysis.txt did not need new dependencies.

Windows path-length limits prevented JupyterLab installation in a deeply nested first test environment. Verification used a fresh short-path temporary venv instead, without changing Windows settings. Keep environment paths short if this occurs on another Windows machine.

## Limits

Full-mode raw-data loading and teacher reevaluation were not performed during this notebook check. The missing-data guard was tested; teacher inference had already been verified separately. No student or teacher training ran. Historical CSV/JSON teacher results and checkpoint weights were not modified.

## Run locally

Install requirements-analysis.txt, select its Python as the kernel, and use Restart Kernel and Run All in notebooks/KD project notebook.ipynb. Keep RUN_MODE='analysis', ALLOW_DOWNLOADS=False and REEVALUATE_TEACHER=False. The README lists all saved artifacts needed by a partner.

Run analysis tests:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_teacher_analysis.py -v
```

Machine-readable evidence: results/notebook_analysis_verification.json.
