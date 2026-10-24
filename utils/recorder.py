"""Save per-domain BEST/LAST models, class memory and validation records."""
from pathlib import Path

import numpy as np
import torch
from utils.Continual import metrics
from utils.tools import write_json


class Recorder:
    def __init__(self, output):
        self.output = Path(output)

    def start(self, task):
        self.stage = task + 1
        self.directory = self.output / f'd{self.stage:02d}'
        self.directory.mkdir()
        self.best_epoch = 0

    def epoch(self, epoch, train_losses, val_losses, sizes, validation_loss, early, optimizer):
        if early.counter == 0:
            self.best_epoch = epoch
        record = {'stage': self.stage, 'epoch': epoch,
                  'train_loss': float(np.mean(train_losses)), 'validation_loss': validation_loss,
                  'validation_batch_losses': val_losses, 'validation_batch_sizes': sizes,
                  'best_epoch': self.best_epoch, 'best_validation_loss': early.best_ls,
                  'stale': early.counter, 'learning_rate': optimizer.param_groups[0]['lr']}
        import json
        with (self.directory / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        write_json(self.output / 'progress.json', record)

    def memory_before(self, memory, current, present):
        self.memory_record = {'before': memory.prototypes.detach().cpu().clone(),
                              'initialized_before': memory.initialized.detach().cpu().clone(),
                              'current': current.detach().cpu().clone(),
                              'present': present.detach().cpu().clone(),
                              'momentum': memory.momentum, 'stage': self.stage,
                              'source': 'current-domain train features; LAST model in eval mode'}

    def memory_after(self, memory):
        self.memory_record['after'] = memory.prototypes.detach().cpu().clone()
        self.memory_record['initialized_after'] = memory.initialized.detach().cpu().clone()
        torch.save(self.memory_record, self.directory / 'class_slot_memory.pt')

    def models(self, model, early, config, epoch):
        for name, instance, selected in (('best', early.best_model, self.best_epoch), ('last', model, epoch)):
            torch.save({'format': 'pmsa_v1', 'config': vars(config).copy(), 'epoch': selected,
                        'stage': self.stage, 'role': name,
                        'state_dict': {k: v.detach().cpu() for k, v in instance.state_dict().items()}},
                       self.directory / f'{name}.pt')
        write_json(self.directory / 'summary.json', {
            'epochs_ran': epoch, 'best_epoch': self.best_epoch,
            'best_validation_loss': early.best_ls, 'next_stage_model': 'last.pt',
            'evaluation_model': 'best.pt'})

    def matrix(self, matrix):
        write_json(self.output / 'validation_results.json', {
            'split': 'validation', 'accuracy_matrix': matrix, 'metrics': metrics(matrix)})
