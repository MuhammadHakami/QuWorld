"""Control: FORCE/RLS readout-only adaptation (always-on baseline).

Every structure strategy must beat this. Adapts ONLY the model's final linear
readout — the last 1x1 conv of proj (a per-pixel linear map from 64 features
to the output channels), which is the one honestly-linear readout in the
architecture, so classic RLS applies exactly:

  P(0) = I / alpha (alpha=10, FORCE);   per pixel-sample z (features + bias):
  k = P z / (1 + z^T P z);  P <- P - k z^T P;  theta_c <- theta_c - e_c k

with the a-priori error e = model(x_t) - x_{t+1} at the sampled pixels
(errors from the batched forward are a-priori w.r.t. the batch-start theta).
Pixels are subsampled (rls_pixels per step, seeded) to bound CPU cost.
Cost: 1 fwd = 1 unit per step; the RLS algebra is O(65^2) per pixel-sample.
"""

import torch

from _bootstrap import add_paths

add_paths()

from strategies.base import EvalMeter, Strategy  # noqa: E402
from substrate import set_adaptable  # noqa: E402


class RLSStrategy(Strategy):
    name = "rls"
    seed_offset = 50
    groups = ()  # readout-only: does not touch structure params

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last_loss = float("nan")
        self._feat = {}

    def adapt(self, segment, meter: EvalMeter, probe) -> None:
        cfg = self.cfg
        set_adaptable(self.model, ())
        conv = self.model.proj[2]  # Conv2d(64 -> in_ch, kernel 1)
        c_out, c_in = conv.weight.shape[0], conv.weight.shape[1]
        theta = torch.cat([conv.weight.detach().reshape(c_out, c_in),
                           conv.bias.detach().reshape(c_out, 1)], dim=1)
        p_mat = torch.eye(c_in + 1) / cfg.rls_alpha
        hook = self.model.proj[1].register_forward_hook(
            lambda mod, args, out: self._feat.__setitem__("f", out))
        it = segment.batches(cfg.batch, 1)
        try:
            while meter.try_charge(fwd=1):
                x, ys = next(it)
                self.samples_seen += x.shape[0]
                x, y = x.to(self.device), ys[0].to(self.device)
                with torch.no_grad():
                    pred = self.model(x)
                    err = pred - y
                self.last_loss = err.square().mean().item()
                feats = self._feat["f"]                     # (B, 64, H, W)
                b, _, hh, ww = feats.shape
                npix = b * hh * ww
                picks = torch.randint(0, npix, (cfg.rls_pixels,),
                                      generator=self.gen)
                z_all = feats.permute(0, 2, 3, 1).reshape(npix, c_in)[picks]
                e_all = err.permute(0, 2, 3, 1).reshape(npix, c_out)[picks]
                ones = torch.ones(cfg.rls_pixels, 1)
                z_all = torch.cat([z_all, ones], dim=1)     # (m, 65)
                for i in range(cfg.rls_pixels):
                    z = z_all[i]
                    pz = p_mat @ z
                    kgain = pz / (1.0 + z @ pz)
                    p_mat = p_mat - torch.outer(kgain, pz)
                    theta = theta - torch.outer(e_all[i], kgain)
                with torch.no_grad():
                    conv.weight.copy_(theta[:, :c_in].reshape(
                        c_out, c_in, 1, 1))
                    conv.bias.copy_(theta[:, c_in])
                probe(meter.used, self.samples_seen)
        finally:
            hook.remove()

    def diagnostics(self) -> dict:
        return {"last_loss": self.last_loss}
