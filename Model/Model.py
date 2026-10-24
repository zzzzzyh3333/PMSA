"""Multiscale temporal classification with batch prototypes and learned slots.

The original per-variable Conv1D/BatchNorm/GRU encoder is shared across scales.
Prototype routing uses the current batch in BOTH training and evaluation:
predictions therefore depend on batch composition. No historical replay,
teacher network, or raw-sample replay is used by the encoder. The training
helper retains class-conditioned slot feature prototypes across domains.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalBlock(nn.Module):
    """Original single-scale local projection and global temporal aggregation."""

    def __init__(self, local_channels, output_size):
        super().__init__()
        self.local_projection = nn.Sequential(
            nn.Conv1d(1, local_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(local_channels),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=local_channels,
            hidden_size=output_size,
            num_layers=1,
            batch_first=True,
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.local_projection(x).transpose(1, 2)
        x, _ = self.gru(x)
        return self.activation(x)[:, -1, :]


class ScaleBankRC(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # Retained for configuration compatibility, never used to infer an axis.
        self.seq_len = getattr(configs, "seq_len", None)
        self.num_variables = configs.enc_in
        self.fusion = getattr(configs, "fusion", "concat")
        self.input_layout = getattr(configs, "input_layout", "BCT").upper()
        self.scales = tuple(getattr(configs, "scales", (1, 2, 3, 4, 5, 6)))
        self.num_slots = getattr(configs, "num_slots", 8)
        self.slot_dim = getattr(configs, "slot_dim", 128)
        self.temperature = getattr(configs, "temperature", 0.1)

        if self.input_layout not in ("BCT", "BTC"):
            raise ValueError("input_layout must be BCT or BTC")
        if not self.scales or any(type(s) is not int or s < 1 for s in self.scales):
            raise ValueError("scales must be a nonempty sequence of positive integers")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("scales must not contain duplicates")
        if self.num_slots < 2 or self.slot_dim < 1:
            raise ValueError("num_slots must be >= 2 and slot_dim must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if min(configs.temporal_channels, configs.temporal_size,
               self.num_variables, configs.num_class) < 1:
            raise ValueError("encoder dimensions and num_class must be positive")

        self.time = _TemporalBlock(configs.temporal_channels, configs.temporal_size)
        if self.fusion == "concat":
            fusion_size = self.num_variables * configs.temporal_size
        elif self.fusion == "sum":
            fusion_size = configs.temporal_size
        else:
            raise ValueError("fusion must be concat or sum")

        self.scale_projection = nn.Sequential(
            nn.Linear(fusion_size, self.slot_dim),
            nn.LayerNorm(self.slot_dim),
        )
        self.slots = nn.Parameter(torch.empty(self.num_slots, self.slot_dim))
        nn.init.normal_(self.slots, std=self.slot_dim ** -0.5)
        self.predictor = nn.Linear(self.num_slots * self.slot_dim, configs.num_class)

    def _to_bct(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected a 3-D input, received {tuple(x.shape)}")
        if self.input_layout == "BTC":
            x = x.transpose(1, 2)
        if any(size == 0 for size in x.shape):
            raise ValueError("Batch, variable, and time dimensions must be nonempty")
        if self.fusion == "concat" and x.size(1) != self.num_variables:
            raise ValueError(
                f"Expected {self.num_variables} variables with layout "
                f"{self.input_layout}, received shape {tuple(x.shape)} after conversion"
            )
        return x

    def _encode_variables_bct(self, x):
        batch_size, num_variables, seq_len = x.shape
        x = x.reshape(batch_size * num_variables, 1, seq_len)
        # Keep the original encoder's output activation.
        x = torch.sigmoid(self.time(x))
        return x.reshape(batch_size, num_variables, -1)

    def encode_variables(self, x):
        """Single-scale compatibility API: return [B, C, temporal_size]."""
        return self._encode_variables_bct(self._to_bct(x))

    def encode_scales(self, x):
        """Return [B, S, D] using parallel pools and one shared encoder."""
        x = self._to_bct(x)
        if x.size(-1) < max(self.scales):
            raise ValueError(
                f"Time length {x.size(-1)} is shorter than largest pool {max(self.scales)}"
            )
        features = []
        for scale in self.scales:
            pooled = x if scale == 1 else F.avg_pool1d(x, scale, stride=scale)
            variables = self._encode_variables_bct(pooled)
            if self.fusion == "concat":
                fused = variables.flatten(start_dim=1)
            else:
                fused = variables.sum(dim=1)
            features.append(self.scale_projection(fused))
        return torch.stack(features, dim=1)

    def align_scales(self, features):
        """Route sample features using batch prototypes, without storing history."""
        prototypes = features.mean(dim=0)  # [S, D]
        similarity = F.normalize(prototypes, dim=-1) @ F.normalize(self.slots, dim=-1).T
        log_assignments = F.log_softmax(similarity / self.temperature, dim=-1)
        assignments = log_assignments.exp()  # [S, J], sums to one over slots
        # Exactly A[s,j] / sum_s A[s,j], calculated in log space for stability.
        aggregation_weights = F.softmax(log_assignments, dim=0)
        slot_features = torch.einsum("bsd,sj->bjd", features, aggregation_weights)
        return slot_features, prototypes, assignments

    def slot_contrastive_loss(self, slot_features):
        contents = slot_features.mean(dim=0)  # [J, D]
        contrast_logits = (
            F.normalize(contents, dim=-1) @ F.normalize(self.slots, dim=-1).T
        ) / self.temperature
        targets = torch.arange(self.num_slots, device=contents.device)
        return F.cross_entropy(contrast_logits, targets)

    def forward(self, x, y=None, return_aux=False):
        del y  # Labels never participate in routing, including at evaluation.
        features = self.encode_scales(x)
        slot_features, prototypes, assignments = self.align_scales(features)
        logits = self.predictor(slot_features.flatten(start_dim=1))
        if not return_aux:
            return logits
        return logits, {
            "slot_loss": self.slot_contrastive_loss(slot_features),
            "scale_features": features,
            "scale_prototypes": prototypes,
            "assignments": assignments,
            "slot_features": slot_features,
            "slot_contents": slot_features.mean(dim=0),
        }


class ClassSlotRelationConsolidation(nn.Module):
    """Maintain class-slot prototypes and preserve them across domains."""

    def __init__(self, num_classes, num_slots, slot_dim, momentum=0.9):
        super().__init__()
        if min(num_classes, num_slots, slot_dim) < 1:
            raise ValueError("class-slot memory dimensions must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, num_slots, slot_dim)
        )
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    def forward(self, slot_features, labels):
        if slot_features.ndim != 3:
            raise ValueError("slot_features must have shape [B, J, D]")
        if labels.ndim != 1 or labels.size(0) != slot_features.size(0):
            raise ValueError("labels must have shape [B]")

        losses = []
        for class_tensor in labels.unique():
            class_index = int(class_tensor.item())
            if class_index < 0 or class_index >= self.num_classes:
                raise ValueError("class label is outside the class-slot memory")
            if not self.initialized[class_index]:
                continue
            class_center = slot_features[labels == class_tensor].mean(dim=0)
            reference = self.prototypes[class_index].detach()
            losses.append((1.0 - F.cosine_similarity(class_center, reference, dim=-1)).mean())

        if not losses:
            return slot_features.sum() * 0.0
        return torch.stack(losses).mean()

    @torch.no_grad()
    def update(self, current_prototypes, present_classes):
        if current_prototypes.shape != self.prototypes.shape:
            raise ValueError("current class-slot prototypes have an invalid shape")
        if present_classes.shape != self.initialized.shape:
            raise ValueError("present_classes has an invalid shape")

        new_classes = present_classes & ~self.initialized
        known_classes = present_classes & self.initialized
        self.prototypes[new_classes] = current_prototypes[new_classes]
        # Boolean indexing returns a copy; explicitly assign the EMA back.
        self.prototypes[known_classes] = (
            self.momentum * self.prototypes[known_classes]
            + (1.0 - self.momentum) * current_prototypes[known_classes]
        )
        self.initialized[present_classes] = True
