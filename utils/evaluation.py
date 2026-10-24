"""Evaluate frozen stage checkpoints on one explicitly selected split."""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from Model import make_model
from utils.Continual import metrics


@torch.no_grad()
def evaluate(model, domain, config):
    model.eval()
    k = config.num_class
    cm = torch.zeros(k, k, dtype=torch.int64)
    loss_sum, count = 0.0, 0
    x, y = domain['samples'], domain['labels']
    for start in range(0, len(y), config.eval_batchsize):
        xb = x[start:start + config.eval_batchsize].float().to(config.device)
        yb = y[start:start + config.eval_batchsize].long().to(config.device)
        logits = model(xb)
        if logits.shape != (len(yb), k) or not torch.isfinite(logits).all():
            raise ValueError('Invalid model output')
        loss_sum += F.cross_entropy(logits, yb, reduction='sum').item()
        count += len(yb)
        cm += torch.bincount((yb * k + logits.argmax(-1)).cpu(), minlength=k*k).reshape(k, k)
    return {'accuracy': cm.trace().item() / count, 'loss': loss_sum / count,
            'examples': count, 'confusion_matrix': cm.tolist()}


def evaluate_stream(run, domains, config):
    matrix, details = [], []
    for stage in range(1, config.tasks + 1):
        checkpoint = Path(run) / f'd{stage:02d}/best.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['stage'] != stage or payload['role'] != 'best':
            raise ValueError(f'Unexpected checkpoint role or stage: {checkpoint}')
        saved = dict(payload['config'])
        # Runtime device can change; all model/evaluation settings come from the run.
        saved['device'] = config.device
        stage_config = SimpleNamespace(**saved)
        model = make_model(stage_config).float().to(config.device)
        model.load_state_dict(payload['state_dict'], strict=True)
        row = [evaluate(model, domain, stage_config) for domain in domains[:stage]]
        matrix.append([item['accuracy'] for item in row])
        details.append(row)
    return {'accuracy_matrix': matrix, 'details': details, 'metrics': metrics(matrix)}
