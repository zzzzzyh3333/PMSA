"""PMSA for HAR with one linear classifier; historical parameter keys retained."""
from torch import nn
from .Model import ScaleBankRC, ClassSlotRelationConsolidation


class PMSA(ScaleBankRC):
    """Single affine classifier with the experiment's input-dropout hook."""

    def __init__(self, config):
        super().__init__(config)
        self.head_dropout = nn.Dropout(getattr(config, 'head_dropout', 0.0))
        self.predictor.register_forward_pre_hook(self._before_classifier)

    def _before_classifier(self, module, inputs):
        return (self.head_dropout(inputs[0]),)


def make_model(config):
    head = getattr(config, 'classifier_head', 'single_linear')
    if head != 'single_linear':
        raise ValueError('This package provides only the single-linear classifier')
    return PMSA(config)
