"""Experimental causal, sparse signal-to-register routing.

The score predicts TRAINING-defined classical error exceedance, not intrinsic
'quantum hardness'. Spatial detail alone is not evidence of quantum usefulness.
This module is isolated from the frozen configuration-screen implementation.
"""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from config_search.core import ConfigurableCore, RegisterConfig
from model import MLPCore, _count


class SignalRouter(nn.Module):
    def __init__(self, channels: int, config: RegisterConfig, arm='quantum', threshold=.5):
        super().__init__()
        if arm not in ('quantum', 'classical'):
            raise ValueError('Need quantum or architecture-matched classical control')
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Threshold must be in [0,1]')
        self.channels, self.config, self.arm = channels, config, arm
        self.register_buffer('threshold', torch.tensor(float(threshold)))
        self.score = nn.Sequential(nn.Linear(channels + 2, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        cores, targets = [], []
        for _ in range(config.registers):
            single = replace(config, registers=1)
            # Consume the same reference RNG in both arms; classical-core
            # initialization must not perturb shared router/projection weights.
            reference = ConfigurableCore(channels, single)
            target = _count(reference)
            if arm == 'classical':
                with torch.random.fork_rng(devices=[]):
                    core = MLPCore(channels, config.width, target)
            else:
                core = reference
            cores.append(core); targets.append(target)
        self.cores = nn.ModuleList(cores)
        self.projections = nn.ModuleList([nn.Linear(config.width, channels) for _ in cores])
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.core_parameter_gaps = [_count(c) - t for c, t in zip(cores, targets)]
        self.warmup = False
        self.register_budget = None
        self.last_route = {}

    def features(self, x):
        if x.ndim < 2 or x.shape[1] != self.channels:
            raise ValueError('Expected [batch, channels, ...] backbone latent')
        tokens = x.reshape(x.shape[0], self.channels, -1).transpose(1, 2)
        normalized = F.layer_norm(tokens, (self.channels,))
        amplitude = tokens.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        detail = (tokens - tokens.mean(1, keepdim=True)).square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        features = torch.cat([normalized, amplitude.log1p(), detail.log1p()], -1)
        return tokens, normalized, features

    def difficulty_logits(self, x):
        return self.score(self.features(x)[2]).squeeze(-1)

    def difficulty_loss(self, x, error_exceeds_threshold):
        """Labels must come from training-only classical prediction errors.

        Caller defines/resizes the error map and freezes its threshold; inference
        never supplies targets. This is a difficulty proxy, not benefit labels.
        """
        logits = self.difficulty_logits(x)
        labels = error_exceeds_threshold.detach().to(logits)
        if labels.shape != logits.shape or not torch.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
            raise ValueError('Require finite [batch, tokens] targets in [0,1]')
        return F.binary_cross_entropy_with_logits(logits, labels)

    def residual_update(self, slot, readout, raw_tokens):
        return self.alpha * self.projections[slot](readout)

    def choose_locations(self, probabilities, slots):
        return probabilities.topk(slots, dim=1, sorted=True)

    def forward(self, x, *, budget=None):
        tokens, normalized, features = self.features(x)
        probabilities = self.score(features).squeeze(-1).sigmoid()
        slots = min(self.config.registers, tokens.shape[1])
        values, indices = self.choose_locations(probabilities, slots)
        active = values > self.threshold
        if self.training and self.warmup:
            active = torch.ones_like(active)
        budget = self.register_budget if budget is None else budget
        if budget is not None:
            cap = torch.as_tensor(budget, device=x.device)
            if cap.dtype == torch.bool or cap.is_floating_point() or cap.is_complex():
                raise ValueError('Register budget must be an integer')
            if cap.ndim > 1 or (cap.ndim == 1 and len(cap) != len(x)):
                raise ValueError('Budget must be scalar or one integer per input')
            if ((cap < 0) | (cap > self.config.registers)).any():
                raise ValueError('Register budget outside available capacity')
            cap = cap.expand(len(x))
            active = active & (torch.arange(slots, device=x.device)[None, :] < cap[:, None])
        output = tokens.clone()
        calls = 0
        for slot in range(slots):
            batch_ids = active[:, slot].nonzero(as_tuple=True)[0]
            if batch_ids.numel() == 0:
                continue  # Actual conditional execution, not multiply-after-compute.
            token_ids = indices[batch_ids, slot]
            h = normalized[batch_ids, token_ids]
            readout = self.cores[slot](h)
            update = self.residual_update(slot, readout, tokens[batch_ids, token_ids])
            update = update * values[batch_ids, slot, None]
            output[batch_ids, token_ids] = output[batch_ids, token_ids] + update
            calls += 1
        self.last_route = dict(indices=indices.detach(), probabilities=values.detach(),
                               active=active.detach(), active_qubits=active.sum(1).detach() * self.config.width,
                               executed_register_batches=calls, warmup=bool(self.training and self.warmup))
        return output.transpose(1, 2).reshape_as(x)


class RelativeSignalRouter(SignalRouter):
    """Bound each selected-token correction norm by scale * token norm.

    Distinct selected locations prevent accumulation of multiple corrections at
    one token. This bounds this adapter only, not downstream rollout error.
    """
    def __init__(self, *args, relative_scale=.1, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(relative_scale) or not 0 < relative_scale <= 1:
            raise ValueError('Require finite relative scale in (0,1]')
        self.relative_scale = relative_scale

    def residual_update(self, slot, readout, raw_tokens):
        projected = self.projections[slot](readout)
        direction = F.layer_norm(projected, (self.channels,)).tanh()
        rms = torch.linalg.vector_norm(raw_tokens, dim=-1, keepdim=True) / math.sqrt(self.channels)
        return self.relative_scale * rms * self.alpha.tanh() * direction


class ControlledSignalRouter(RelativeSignalRouter):
    """Causal location controls; no mixing between independent batch members."""
    def __init__(self, *args, routing='adaptive', **kwargs):
        super().__init__(*args, **kwargs)
        if routing not in ('adaptive', 'static', 'permuted'):
            raise ValueError('Unknown routing control')
        self.routing = routing
        self._permutations = {}

    def choose_locations(self, probabilities, slots):
        if self.routing == 'static':
            indices = torch.linspace(0, probabilities.shape[1]-1, slots,
                                     device=probabilities.device).long()
            indices = indices[None, :].expand(len(probabilities), -1)
            return probabilities.gather(1, indices), indices
        values, indices = super().choose_locations(probabilities, slots)
        if self.routing == 'permuted':
            key = (probabilities.shape[1], probabilities.device)
            if key not in self._permutations:
                generator = torch.Generator().manual_seed(90417)
                self._permutations[key] = torch.randperm(key[0], generator=generator).to(key[1])
            indices = self._permutations[key][indices]
        return values, indices
