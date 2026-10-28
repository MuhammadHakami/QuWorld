"""Doc 35 §46: batched GPU free-boundary Grad-Shafranov solver (Picard iteration, coil currents fixed).

psi = psi_coil + Gpp @ (J dA). Each iteration (phi = sign(Ip) * psi, so the plasma is a maximum of phi):
  axis   = the strict 3 x 3 local maximum of phi inside the limiter nearest the current centroid;
  bound  = max over limiter-excluded pixels of reach(phi) (the max-min path value from the axis): the highest level at which the
           axis-connected region {phi > level} first touches the limiter or an X-point leg leaving it -> the separatrix level for a
           diverted plasma; plasma = {reach > bound};
  J      = L [b0 R / R0 + (1 - b0) R0 / R] (1 - psiN^am)^an on the plasma, L from Ip (FreeGS-style profile; ideas only);
  vertical position: the coil field's radial component makes the vertical equilibrium unstable for Picard; J is shifted rigidly in
           Z to the force-free position F_z = -2 pi int R J B_R,coil dA = 0 (linearised, clipped to half a cell per step);
  relax  J <- (1 - w) J + w J_new.
Returns psi (B, nZ, nR) in Wb/rad (physical sign), J, axis, boundary level and convergence diagnostics.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class Solver:
    def __init__(self, m, device="cuda", dtype=torch.float32):
        self.m, self.dev, self.dt = m, torch.device(device), dtype
        self.nZ, self.nR = len(m.Z), len(m.R)
        self.R = torch.tensor(m.R, device=self.dev, dtype=dtype); self.Z = torch.tensor(m.Z, device=self.dev, dtype=dtype)
        self.RR = self.R[None, :].expand(self.nZ, self.nR); self.ZZ = self.Z[:, None].expand(self.nZ, self.nR)
        self.dR, self.dZ = float(m.R[1] - m.R[0]), float(m.Z[1] - m.Z[0]); self.dA = self.dR * self.dZ
        self.Gc = torch.tensor(m.Gc, device=self.dev, dtype=torch.float64)
        self.Gpp = torch.tensor(m.Gpp, device=self.dev, dtype=dtype)
        lim = torch.tensor(m.limiter, device=self.dev)
        self.lim = lim; self.out = ~lim
        er = lim.clone()
        e = er.clone(); e[1:] &= er[:-1]; e[:-1] &= er[1:]; e[:, 1:] &= er[:, :-1]; e[:, :-1] &= er[:, 1:]
        self.inner = e

    # ---------------------------------------------------------------- pieces
    def coil_psi(self, I_kA: torch.Tensor) -> torch.Tensor:
        return (I_kA.to(self.dev, torch.float64) @ self.Gc).view(-1, self.nZ, self.nR).to(self.dt)

    def plasma_psi(self, J: torch.Tensor) -> torch.Tensor:
        return ((J.flatten(1) * self.dA) @ self.Gpp).view(-1, self.nZ, self.nR)        # Gpp symmetric

    def reach(self, phi, iz, ir, h0=None):
        B = phi.shape[0]; b = torch.arange(B, device=self.dev)
        h = torch.full_like(phi, -np.inf) if h0 is None else torch.minimum(h0, phi)
        h[b, iz, ir] = phi[b, iz, ir]
        for it in range(600):
            g = torch.minimum(torch.maximum(F.max_pool2d(h[:, None], (3, 1), 1, (1, 0)), F.max_pool2d(h[:, None], (1, 3), 1, (0, 1)))[:, 0], phi)
            if it % 8 == 7 and torch.equal(g, h):
                break
            h = g
        return h

    def axis(self, phi, Rc, Zc):
        p = torch.where(self.inner, phi, torch.full_like(phi, -np.inf))
        nb = F.max_pool2d(F.pad(p[:, None], (1, 1, 1, 1), value=-np.inf), 3, 1)[:, 0]
        lm = (p >= nb) & self.inner & torch.isfinite(p)
        d2 = (self.RR[None] - Rc[:, None, None]) ** 2 + (self.ZZ[None] - Zc[:, None, None]) ** 2
        score = torch.where(lm, d2, torch.full_like(d2, np.inf)).flatten(1)
        has = torch.isfinite(score).any(1)
        idx = torch.where(has, score.argmin(1), p.flatten(1).argmax(1))
        return idx // self.nR, idx % self.nR, has

    def profile(self, phi, h, iz, ir, Ip, b0, am, an):
        B = phi.shape[0]; b = torch.arange(B, device=self.dev)
        pax = phi[b, iz, ir]
        bnd = torch.where(self.out[None], h, torch.full_like(h, -np.inf)).flatten(1).max(1).values
        P = (h > bnd[:, None, None]) & self.lim[None]
        psin = ((pax[:, None, None] - phi) / (pax - bnd).clamp(min=1e-9)[:, None, None]).clamp(0, 1)
        R0 = self.m.R0; Rf = self.RR[None]
        shape = (b0[:, None, None] * Rf / R0 + (1 - b0[:, None, None]) * R0 / Rf) * (1 - psin ** am[:, None, None]).clamp(min=0) ** an[:, None, None]
        J = torch.where(P, shape, torch.zeros_like(shape))
        tot = J.sum((1, 2)) * self.dA
        J = J * (Ip.abs() / tot.clamp(min=1e-12))[:, None, None] * torch.sign(Ip)[:, None, None]
        return J, P, pax, bnd

    def r_axis(self, phi, iz, ir):
        """Magnetic-axis R with 3-point parabolic refinement."""
        b = torch.arange(phi.shape[0], device=self.dev); irm, irp = (ir - 1).clamp(0, self.nR - 1), (ir + 1).clamp(0, self.nR - 1)
        a, c0, c = phi[b, iz, irm], phi[b, iz, ir], phi[b, iz, irp]; den = a - 2 * c0 + c
        dr = torch.where(den.abs() > 1e-12, (0.5 * (a - c) / den).clamp(-1, 1), torch.zeros_like(den))
        return self.R[ir] + dr * self.dR

    def r_out(self, phi, P, iz, bnd):
        """Outer LCFS crossing on the axis row, interpolated to the boundary level."""
        b = torch.arange(phi.shape[0], device=self.dev); row = P[b, iz]                 # (B, nR)
        idx = torch.arange(self.nR, device=self.dev).expand_as(row)
        io = torch.where(row, idx, torch.full_like(idx, -1)).max(1).values.clamp(0, self.nR - 2)
        p0 = phi[b, iz, io]; p1 = phi[b, iz, io + 1]
        f = ((p0 - bnd) / (p0 - p1).clamp(min=1e-12)).clamp(0, 1)
        return torch.where(row.any(1), self.R[io] + f * self.dR, torch.full_like(p0, float("nan")))

    def vshift(self, J, BRc, dBRdz, frac=0.5):
        Rf = self.RR[None]
        Fz = -(2 * np.pi * Rf * J * BRc).sum((1, 2)); dF = -(2 * np.pi * Rf * J * dBRdz).sum((1, 2))
        dz = (-Fz / torch.where(dF.abs() > 1e-12, dF, torch.full_like(dF, 1e-12))).clamp(-frac * self.dZ, frac * self.dZ)   # F(dz) = Fz + dz dF = 0
        dz = torch.where(torch.isfinite(dz), dz, torch.zeros_like(dz))
        s = dz / self.dZ                                                       # shift J by +dz: J'(Z) = J(Z - dz)
        Jm = torch.roll(J, 1, 1); Jp = torch.roll(J, -1, 1); Jm[:, 0] = 0; Jp[:, -1] = 0
        sp, sn = s.clamp(min=0)[:, None, None], (-s).clamp(min=0)[:, None, None]
        return (1 - sp - sn) * J + sp * Jm + sn * Jp, Fz

    # ---------------------------------------------------------------- solve
    @torch.no_grad()
    def solve(self, I_kA, Ip_kA, b0=0.5, am=1.0, an=2.0, n_iter=80, w=0.4, R_init=None, Z_init=0.0, width=0.25, vertical=True, tol=1e-4,
              pc=None, J0=None, R_out=None, gain=0.3, kp=0.1, ki=0.01, gt=0.3, R_ax=None, gb=2.0, b0_lim=(-1.0, 2.0)):
        """pc: optional external flux (B, nZ, nR) replacing the coil tables (validation with EFIT's own external field).
        R_out: optional target outer-midplane LCFS radius (B,) [m]; a uniform vertical-field correction Bv (psi += Bv R^2 / 2, the field
        of distant / unmeasured currents) is fed back each iteration to put the LCFS there (radial position control)."""
        B = len(Ip_kA); dev = self.dev
        Ip = torch.as_tensor(Ip_kA, device=dev, dtype=self.dt) * 1e3
        par = lambda v: torch.as_tensor(v, device=dev, dtype=self.dt).expand(B).clone()  # noqa: E731
        b0, am, an = par(b0), par(am), par(an)
        pc = self.coil_psi(I_kA) if pc is None else torch.as_tensor(pc, device=dev, dtype=self.dt)
        BRc = -torch.gradient(pc, spacing=(self.Z,), dim=1)[0] / self.RR[None]
        dBRdz = torch.gradient(BRc, spacing=(self.Z,), dim=1)[0]
        Rc = par(R_init if R_init is not None else self.m.R0); Zc = par(Z_init)
        J = torch.exp(-((self.RR[None] - Rc[:, None, None]) ** 2 + (self.ZZ[None] - Zc[:, None, None]) ** 2) / (2 * width ** 2)) * self.lim[None]
        J = J * (Ip.abs() / (J.sum((1, 2)) * self.dA))[:, None, None] * torch.sign(Ip)[:, None, None]
        if J0 is not None:
            J = torch.as_tensor(J0, device=dev, dtype=self.dt).clone()
        sg = torch.sign(Ip)[:, None, None]; h = None; hist = []
        Bv = torch.zeros(B, device=dev, dtype=self.dt); R2h = (self.RR ** 2 / 2)[None]
        cz = torch.zeros(B, device=dev, dtype=self.dt); R2Z = (self.RR ** 2 * self.ZZ)[None]            # vacuum radial field c R^2 Z
        cI = torch.zeros(B, device=dev, dtype=self.dt)
        Zt = par(Z_init)
        Rt = None if R_out is None else torch.as_tensor(R_out, device=dev, dtype=self.dt)
        Ra_t = None if R_ax is None else torch.as_tensor(R_ax, device=dev, dtype=self.dt)      # target magnetic-axis R -> profile b0
        for k in range(n_iter):
            psi = pc + Bv[:, None, None] * R2h + cz[:, None, None] * R2Z + self.plasma_psi(J); phi = sg * psi
            tot = J.abs().sum((1, 2)).clamp(min=1e-12)
            Rc = (J.abs() * self.RR[None]).sum((1, 2)) / tot; Zc = (J.abs() * self.ZZ[None]).sum((1, 2)) / tot
            iz, ir, has = self.axis(phi, Rc, Zc)
            h = self.reach(phi, iz, ir)
            Jn, P, pax, bnd = self.profile(phi, h, iz, ir, Ip, b0, am, an)
            if Ra_t is not None:                     # Shafranov shift: more pressure-like current (b0) moves the axis outward
                ra = self.r_axis(phi, iz, ir)
                b0 = (b0 - gb * torch.where(torch.isfinite(ra), ra - Ra_t, torch.zeros_like(ra))).clamp(*b0_lim)
            if Rt is not None:
                ro = self.r_out(phi, P, iz, bnd)
                Bv = Bv - gain * torch.sign(Ip) * torch.where(torch.isfinite(ro), ro - Rt, torch.zeros_like(ro))
            ok = P.flatten(1).sum(1) >= 8
            Jn = torch.where(ok[:, None, None], Jn, J)
            if vertical:                     # hold the centroid at Zt with c R^2 Z; move Zt to the coil-field force-free height
                Rf = self.RR[None]; Fz = -(2 * np.pi * Rf * J * BRc).sum((1, 2)); dF = -(2 * np.pi * Rf * J * dBRdz).sum((1, 2))
                dzf = (-Fz / torch.where(dF.abs() > 1e-12, dF, torch.full_like(dF, 1e-12))).clamp(-0.05, 0.05)
                Zt = Zt + gt * torch.where(torch.isfinite(dzf), dzf, torch.zeros_like(dzf))
                e = Zc - Zt; cI = cI - ki * torch.sign(Ip) * e; cz = cI - kp * torch.sign(Ip) * e      # PI, T/m^2 (stiffness ~0.03 / m)
            dJ = ((Jn - J).abs().sum((1, 2)) / tot)
            J = (1 - w) * J + w * Jn; hist.append(float(dJ.median()))
            if k > 10 and float(dJ.max()) < tol:
                break
        psi = pc + Bv[:, None, None] * R2h + cz[:, None, None] * R2Z + self.plasma_psi(J)
        return dict(b0=b0, cz=cz, Zt=Zt, Bv=Bv, psi=psi, J=J, iz=iz, ir=ir, bnd=bnd, pax=pax, ok=ok & has, dJ=dJ, hist=hist, n_iter=k + 1, P=P)
