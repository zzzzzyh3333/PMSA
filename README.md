# PMSA — HAR

Source code for PMSA: shared multiscale temporal encoding, prototype-guided scale alignment, and class-conditioned preservation for continual time-series classification.

This source package contains the **HAR model with a single linear classifier**, sequential training, frozen-checkpoint evaluation, and the original HAR_T ten-domain configuration. Datasets and pretrained weights are not bundled.

## Repository structure

```text
PMSA/
├── Model/
│   ├── Model.py              # Single-linear model and class-conditioned memory
│   ├── __init__.py           # Public PMSA interface and head-input dropout
│   └── run.py                # Check, train, evaluate
├── utils/
│   ├── training.py           # Sequential Trainer, early stopping, memory update
│   ├── data.py               # Native-length domain loading and checks
│   ├── evaluation.py         # BEST-checkpoint evaluation and confusion matrices
│   ├── recorder.py           # BEST/LAST checkpoints and training logs
│   ├── Continual.py          # ACC, RF, PRF
│   └── tools.py              # Seeding, JSON, SHA256
├── configs/HAR.json          # Original HAR_T ten-domain configuration
├── data/README.md            # Required tensor format
├── requirements.txt
└── README.md
```

## Environment

The packaging checks use Python 3.12, PyTorch 2.6.0+cu124 and NumPy 2.0.1. Historical formal runs also used PyTorch 2.7.1+cu128; identical scores across software or hardware versions are not guaranteed. Install the PyTorch build suitable for your machine, then the listed dependencies:

```bash
python -m pip install -r requirements.txt
python Model/run.py --help
python Model/run.py check --config configs/HAR.json --device cpu
```

The check uses synthetic data and does not report research accuracy. CPU is supported for checks and small examples; use an explicitly selected CUDA device for full experiments.

## Model

Each input is a tensor `[batch, channels, time]`. All samples in a domain have the same native length; different domains may have different lengths. The model receives waveforms, without sampling-rate or duration metadata. It applies parallel average pooling at the configured factors, a shared Conv1D/BatchNorm/GRU encoder, channel concatenation and projection, prototype-guided scale alignment, and a linear classifier.

```python
import json
from types import SimpleNamespace
import torch
from Model import make_model

config = SimpleNamespace(**json.load(open("configs/HAR.json")))
model = make_model(config)
x = torch.randn(8, config.enc_in, 128)
logits, auxiliary = model(x, return_aux=True)
```

The classifier is one `nn.Linear(8*128, 6)` layer. The historical implementation name `ScaleBankRC` and parameter keys are retained inside `Model.py` for checkpoint compatibility. Checkpoint loading is strict.

Routing uses batch-mean scale features in both training and inference, followed by normalization over prototypes and over scales. Predictions therefore depend on batch composition. Keep evaluation batch size and sample order fixed. The core retains the original ReLU followed by sigmoid in the temporal encoder.

Training uses classification cross-entropy plus the prototype contrastive and class-conditioned preservation losses. The memory stores class-by-prototype mean feature vectors, updated once per domain by EMA from the LAST model on the training split. It does not store raw examples.

## HAR configuration

| Config | Data protocol | Domains | Head | Prototypes | Train/eval batch | LR | Head dropout | Validation CE |
|---|---|---:|---|---:|---|---:|---:|---|
| `HAR.json` | Original HAR_T, rate changes | 10 | single | 8 | 256/256 | 0.001 | 0 | batch mean |

Pooling factors are `[1,2,3,4,5,6]`, feature dimension is 128, temperature is 0.1, and EMA momentum is 0.9. Both auxiliary loss weights are 0.1. The configuration includes the native domain lengths `[128,64,103,64,116,52,90,52,77,77]` and expected data hashes.

The original HAR_T split inherits overlapping half-windows and is a development/reproduction protocol, not a subject-disjoint confirmation protocol.

## Training and evaluation

Place the prepared `train.pt`, `val.pt`, and `test.pt` in one directory; see [data format](data/README.md). No data download or reconstruction is performed by the runner.

```bash
python Model/run.py train --config configs/HAR.json --data-root data/HAR --output runs/HAR_seed2025 --seed 2025 --device cuda
python Model/run.py evaluate --run runs/HAR_seed2025 --data-root data/HAR --split test --device cuda
```

Repeat with seeds 2026 and 2027 and distinct output directories for the historical three-seed convention. Report test ACC/RF/PRF as the mean and sample standard deviation across all three runs. Validation is for model selection; its scores are stored separately.

Training reads only `train.pt` and `val.pt`. It uses Adam with zero weight decay, ReduceLROnPlateau (factor 0.5, patience 3), early stopping (patience 15), and at most 100 epochs per domain. The next domain continues from LAST; evaluation loads each stage's validation BEST. The historical extra validation-monitor pass is retained to preserve DataLoader random-number consumption. No AMP, gradient clipping, replay, or teacher distillation is added.

Output contains `config.json`, `run_info.json`, validation records and, for every stage, `best.pt`, `last.pt`, `class_slot_memory.pt`, and `history.jsonl`. The separate evaluation command writes `test_evaluation.json` with the triangular accuracy matrix, confusion matrices and metrics. Existing nonempty runs and existing evaluation reports are not overwritten. Optimizer-state resume is not implemented.

For your own prepared data, edit a configuration and explicitly use `--allow-custom-data` in both commands. This skips original hash/length matching, records the custom-data flag, and still checks channels, labels, dimensions and finite values. It defines a different data protocol.

## Metrics

All reported metric values are fractions in `[0,1]`. ACC averages the final row over domains. For each old domain, RF uses the highest accuracy before the final stage, computes the positive relative drop, and averages over old domains. PRF additionally weights each relative drop by `1 - final_accuracy`. Single-stage or zero-peak RF/PRF is undefined and is written as JSON `null`.

## Snapshot scope

`Model.py` is a byte-for-byte copy of the preserved single-linear model source. The training file extracts the original Trainer and adds portable data loading, records, finite-value guards, and explicit validation naming. Runtime dependencies are PyTorch and NumPy.

The HAR packaging checks cover synthetic forward/backward passes, standalone CLI operation after relocation, strict checkpoint reload, and read-only input checks across all ten HAR domains. A two-domain/two-epoch synthetic comparison matched the historical Trainer's BEST/LAST tensors, EMA memories and validation matrix exactly on CPU. These are engineering checks, not new full-dataset results.
