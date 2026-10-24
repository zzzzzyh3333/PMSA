# PMSA

Source code for PMSA: shared multiscale temporal encoding, prototype-guided scale alignment, and class-conditioned preservation for continual time-series classification.


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

The packaging checks use Python 3.12, PyTorch 2.6.0+cu124 and NumPy 2.0.1:

```bash
python -m pip install -r requirements.txt
python Model/run.py --help
python Model/run.py check --config configs/HAR.json --device cpu
```

## Metrics

All reported metric values are fractions in `[0,1]`. ACC averages the final row over domains. For each old domain, RF uses the highest accuracy before the final stage, computes the positive relative drop, and averages over old domains. PRF additionally weights each relative drop by `1 - final_accuracy`. Single-stage or zero-peak RF/PRF is undefined and is written as JSON `null`.
