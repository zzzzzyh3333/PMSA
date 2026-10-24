"""PMSA HAR: check, train, or evaluate the single-linear model."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from Model import make_model
from Model.Model import ClassSlotRelationConsolidation
from utils.data import load_splits
from utils.evaluation import evaluate_stream
from utils.tools import configure, sha256, write_json


def read_config(path, device):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    values['device'] = device
    config = SimpleNamespace(**values)
    if config.dataset != 'HAR' or config.classifier_head != 'single_linear':
        raise ValueError('This package supports HAR with a single-linear classifier')
    for name in ('tasks', 'num_class', 'enc_in', 'batchsize', 'eval_batchsize', 'epoches'):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config.weight_decay != 0:
        raise ValueError('This preserved Trainer uses weight_decay=0')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; explicitly select --device cpu if intended')
    return config


def check_model(config):
    configure(config, 2025)
    model = make_model(config).float().to(config.device)
    length = max(max(config.scales), min(config.expected_lengths))
    # A short synthetic batch checks interfaces, not scientific performance.
    length = min(length, max(24, max(config.scales)))
    x = torch.randn(4, config.enc_in, length, device=config.device)
    labels = torch.arange(4, device=config.device) % config.num_class
    logits, aux = model(x, return_aux=True)
    memory = ClassSlotRelationConsolidation(config.num_class, config.num_slots,
                                            config.slot_dim, config.class_slot_momentum).to(config.device)
    loss = (torch.nn.functional.cross_entropy(logits, labels)
            + config.slot_loss_weight * aux['slot_loss']
            + config.class_slot_loss_weight * memory(aux['slot_features'], labels))
    loss.backward()
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise ValueError('Non-finite forward output')
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Missing or non-finite gradient')
    return {'check': 'synthetic_forward_backward', 'passed': True,
            'dataset_config': config.dataset, 'head': config.classifier_head,
            'input_shape': list(x.shape), 'logits_shape': list(logits.shape),
            'parameters': sum(p.numel() for p in model.parameters()),
            'device': config.device, 'torch': str(torch.__version__), 'numpy': np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='Synthetic forward/backward; no dataset needed')
    check.add_argument('--config', type=Path, required=True)
    check.add_argument('--device', default='cpu')
    train = commands.add_parser('train', help='Train using train.pt and val.pt only')
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--data-root', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--seed', type=int, default=2025)
    train.add_argument('--device', default='cuda')
    train.add_argument('--epochs', type=int, help='Explicitly override the configured epoch limit')
    train.add_argument('--allow-custom-data', action='store_true',
                       help='Accept different data hashes/lengths; record as a custom protocol')
    evaluate = commands.add_parser('evaluate', help='Evaluate all frozen BEST checkpoints')
    evaluate.add_argument('--run', type=Path, required=True)
    evaluate.add_argument('--data-root', type=Path, required=True)
    evaluate.add_argument('--split', choices=('test', 'val'), default='test')
    evaluate.add_argument('--device', default='cuda')
    evaluate.add_argument('--allow-custom-data', action='store_true')
    options = parser.parse_args()

    if options.command == 'check':
        print(json.dumps(check_model(read_config(options.config, options.device)), indent=2))
        return

    if options.command == 'train':
        config = read_config(options.config, options.device)
        if options.epochs is not None:
            if options.epochs < 1:
                parser.error('--epochs must be positive')
            config.epoches = options.epochs
        config.seed = options.seed
        if options.output.exists() and any(options.output.iterdir()):
            raise FileExistsError(f'Output must be empty: {options.output}')
        data, hashes = load_splits(options.data_root, config, check_hash=not options.allow_custom_data)
        options.output.mkdir(parents=True, exist_ok=True)
        write_json(options.output / 'config.json', vars(config))
        write_json(options.output / 'run_info.json', {
            'method': 'PMSA', 'data_protocol': config.data_protocol,
            'custom_data': options.allow_custom_data, 'data_sha256_loaded': hashes,
            'test_loaded_during_training': False, 'seed': options.seed,
            'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(ROOT.rglob('*.py'))
                              if p.parts[len(ROOT.parts)] in ('Model', 'utils')},
            'python': sys.version.split()[0], 'torch': str(torch.__version__), 'numpy': np.__version__,
            'device': options.device, 'continuation': 'LAST', 'selection': 'validation CE BEST',
            'optimizer_state_resume_supported': False})
        configure(config, options.seed)
        from utils.training import train as train_stream
        train_stream(config, data, options.output)
        write_json(options.output / 'completion.json', {'complete': True, 'stages': config.tasks,
                   'config_sha256': sha256(options.output / 'config.json'), 'test_evaluated': False})
        print(f'Training complete: {options.output}')
        return

    complete = json.loads((options.run / 'completion.json').read_text(encoding='utf-8'))
    if not complete['complete'] or complete['config_sha256'] != sha256(options.run / 'config.json'):
        raise ValueError('Training must be complete and its configuration unchanged')
    destination = options.run / f'{options.split}_evaluation.json'
    if destination.exists():
        raise FileExistsError(f'Evaluation output already exists: {destination}')
    config = read_config(options.run / 'config.json', options.device)
    configure(config, config.seed)
    data, hashes = load_splits(options.data_root, config, (options.split,), not options.allow_custom_data)
    result = evaluate_stream(options.run, data[options.split], config)
    result.update(split=options.split, data_sha256=hashes, custom_data=options.allow_custom_data,
                  seed=config.seed, eval_batchsize=config.eval_batchsize,
                  checkpoint_sha256={f'd{i:02d}/best.pt': sha256(options.run / f'd{i:02d}/best.pt')
                                     for i in range(1, config.tasks + 1)})
    write_json(destination, result)
    print(json.dumps(result['metrics'], indent=2))
    print(f'Evaluation saved: {destination}')


if __name__ == '__main__':
    main()
