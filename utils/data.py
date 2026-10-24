"""Load native-length domains without normalization, padding or resampling."""
from pathlib import Path

import torch
from utils.tools import sha256


def load_splits(root, config, splits=('train', 'val'), check_hash=True):
    root = Path(root)
    result, hashes = {}, {}
    for split in splits:
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown split: {split}')
        path = root / (split + '.pt')
        hashes[path.name] = sha256(path)
        expected = config.data_sha256.get(path.name)
        if check_hash and expected and hashes[path.name] != expected:
            raise ValueError(f'{path.name}: SHA256 differs from {config.data_protocol}')
        # Historical datasets contain NumPy arrays; use only trusted .pt files.
        domains = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(domains, (list, tuple)) or len(domains) != config.tasks:
            raise ValueError(f'{split}: expected {config.tasks} domains')
        checked = []
        for i, domain in enumerate(domains):
            x, raw_y = torch.as_tensor(domain['samples']), torch.as_tensor(domain['labels'])
            if raw_y.ndim != 1 or x.ndim != 3 or x.shape[:2] != (len(raw_y), config.enc_in) or not len(raw_y):
                raise ValueError(f'{split} domain {i + 1}: expected samples[N,C,T], labels[N]')
            if not torch.isfinite(x).all() or not torch.isfinite(raw_y).all():
                raise ValueError(f'{split} domain {i + 1}: non-finite data')
            y = raw_y.long()
            if not torch.equal(raw_y, y) or y.min() < 0 or y.max() >= config.num_class:
                raise ValueError(f'{split} domain {i + 1}: labels must be integers in [0, num_class)')
            if x.shape[-1] < max(config.scales):
                raise ValueError(f'{split} domain {i + 1}: sequence shorter than largest pool')
            if check_hash and config.expected_lengths and x.shape[-1] != config.expected_lengths[i]:
                raise ValueError(f'{split} domain {i + 1}: unexpected native length')
            checked.append({'samples': x, 'labels': y})
        result[split] = checked
    return result, hashes
