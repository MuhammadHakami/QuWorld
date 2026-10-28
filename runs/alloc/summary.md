# Allocation-strategy study: frozen trunk + routed co-processor (K = 30 atoms unless stated)

Test metric relative gain vs the frozen trunk (mean over seeds); captured = share of the trunk's weighted error energy removed; U = utilization ceiling of the allocation (share of error energy on the allocated components; captured <= U by `residual_after_band_removal`).


## TRL2D (trunk TEST 0.19359)

| strategy | U | lin gain / captured | mlp gain / captured | ryd gain / captured | ryd vs mlp |
|---|---|---|---|---|---|
| pooled | 0.0556 | +0.06% / 0.0015 (n=3) | +0.06% / 0.0014 (n=3) | +0.06% / 0.0013 (n=3) | +0.01% (2/3, p=0.32) |
| static | 0.2377 | -0.04% / -0.0007 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% (0/3, p=nan) |
| proxy | 0.1719 | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% (0/3, p=nan) |
| learned | 0.2230 | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% (0/3, p=nan) |
| gain | 0.0640 | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% (0/3, p=nan) |
| random | 0.0006 | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% (0/3, p=nan) |
| oracle (diagnostic) | 0.4448 | +0.00% / 0.0000 (n=1) | +0.00% / 0.0000 (n=1) | +0.00% / 0.0000 (n=1) | +0.00% (0/1, p=nan) |

K sweep (captured / U):

| strategy, core | K=10 | K=30 | K=100 | K=300 |
|---|---|---|---|---|
| pooled, mlp | 0.0014 / 0.0556 | 0.0014 / 0.0556 | 0.0011 / 0.0556 | 0.0011 / 0.0556 |
| pooled, ryd | 0.0016 / 0.0556 | 0.0013 / 0.0556 | 0.0011 / 0.0556 | 0.0003 / 0.0556 |
| proxy, mlp | 0.0000 / 0.0745 | 0.0000 / 0.1719 | 0.0000 / 0.3643 | 0.0000 / 0.5941 |
| proxy, ryd | 0.0000 / 0.0745 | 0.0000 / 0.1719 | 0.0000 / 0.3643 | 0.0000 / 0.5941 |

Controls at the VALID-best tile strategy of the register (static):

- ryd vs gutz: +0.00% (1/3, p=0.42)
- ryd vs ryd2d: +0.03% (2/3, p=0.18)
- ryd vs mlpX4: +0.00% (0/3, p=nan)

## rayleigh_benard (trunk TEST 0.64010)

| strategy | U | lin gain / captured | mlp gain / captured | ryd gain / captured | ryd vs mlp |
|---|---|---|---|---|---|
| pooled | 0.0826 | +2.75% / 0.0506 (n=3) | +6.00% / 0.0689 (n=3) | +6.69% / 0.0732 (n=3) | +0.73% (3/3, p=0.02) |
| static | 0.1817 | +28.82% / 0.1121 (n=3) | +33.02% / 0.1352 (n=3) | +32.86% / 0.1317 (n=3) | -0.24% (1/3, p=0.64) |
| proxy | 0.1605 | +28.75% / 0.0873 (n=3) | +36.55% / 0.1173 (n=3) | +36.25% / 0.1130 (n=3) | -0.48% (1/3, p=0.31) |
| learned | 0.2296 | +28.00% / 0.1299 (n=3) | +36.30% / 0.1691 (n=3) | +34.47% / 0.1603 (n=3) | -2.85% (1/3, p=0.27) |
| gain | 0.2265 | +29.46% / 0.1366 (n=3) | +36.47% / 0.1717 (n=3) | +35.55% / 0.1681 (n=3) | -1.46% (1/3, p=0.34) |
| random | 0.0004 | +0.00% / -0.0007 (n=3) | +0.00% / -0.0005 (n=3) | +0.00% / -0.0005 (n=3) | +0.00% (2/3, p=0.41) |
| oracle (diagnostic) | 0.2915 | +30.19% / 0.1473 (n=1) | +36.48% / 0.1732 (n=1) | +34.45% / 0.1610 (n=1) | -3.21% (0/1, p=nan) |

K sweep (captured / U):

| strategy, core | K=10 | K=30 | K=100 | K=300 |
|---|---|---|---|---|
| pooled, mlp | 0.0704 / 0.0826 | 0.0689 / 0.0826 | 0.0645 / 0.0826 | 0.0661 / 0.0826 |
| pooled, ryd | 0.0732 / 0.0826 | 0.0732 / 0.0826 | 0.0732 / 0.0826 | 0.0721 / 0.0826 |
| proxy, mlp | 0.0492 / 0.0679 | 0.1173 / 0.1605 | 0.1655 / 0.2774 | 0.2167 / 0.3994 |
| proxy, ryd | 0.0438 / 0.0679 | 0.1130 / 0.1605 | 0.1557 / 0.2774 | 0.2104 / 0.3994 |

Controls at the VALID-best tile strategy of the register (proxy):

- ryd vs gutz: +6.78% (2/3, p=0.22)
- ryd vs ryd2d: +1.63% (1/3, p=0.43)
- ryd vs mlpX4: -1.40% (0/3, p=0.07)
- division of labor, hyb (register + MLP) vs mlp2 (two MLPs): -0.43% (2/3, p=0.65)

## shear_flow (trunk TEST 1.05987)

| strategy | U | lin gain / captured | mlp gain / captured | ryd gain / captured | ryd vs mlp |
|---|---|---|---|---|---|
| pooled | 0.0387 | +3.48% / 0.0298 (n=3) | +6.38% / 0.0364 (n=3) | +6.77% / 0.0357 (n=3) | +0.41% (3/3, p=0.09) |
| static | 0.2153 | +22.37% / 0.2083 (n=3) | +22.36% / 0.2087 (n=3) | +22.36% / 0.2082 (n=3) | -0.00% (2/3, p=0.74) |
| proxy | 0.3045 | +26.32% / 0.2836 (n=3) | +26.33% / 0.2856 (n=3) | +26.32% / 0.2843 (n=3) | -0.00% (2/3, p=0.96) |
| learned | 0.3350 | +26.26% / 0.3150 (n=3) | +26.27% / 0.3169 (n=3) | +26.29% / 0.3164 (n=3) | +0.03% (3/3, p=0.00) |
| gain | 0.3324 | +26.29% / 0.3171 (n=3) | +26.22% / 0.3171 (n=3) | +26.26% / 0.3175 (n=3) | +0.06% (3/3, p=0.23) |
| random | 0.0003 | +0.01% / 0.0002 (n=3) | +0.01% / 0.0001 (n=3) | +0.01% / 0.0002 (n=3) | +0.00% (3/3, p=0.36) |
| oracle (diagnostic) | 0.3532 | +26.44% / 0.3310 (n=1) | +26.44% / 0.3314 (n=1) | +26.44% / 0.3312 (n=1) | -0.01% (0/1, p=nan) |

K sweep (captured / U):

| strategy, core | K=10 | K=30 | K=100 | K=300 |
|---|---|---|---|---|
| pooled, mlp | 0.0368 / 0.0387 | 0.0364 / 0.0387 | 0.0360 / 0.0387 | 0.0373 / 0.0387 |
| pooled, ryd | 0.0352 / 0.0387 | 0.0357 / 0.0387 | 0.0365 / 0.0387 | 0.0362 / 0.0387 |
| proxy, mlp | 0.1067 / 0.1147 | 0.2856 / 0.3045 | 0.5842 / 0.6288 | 0.6797 / 0.7624 |
| proxy, ryd | 0.1064 / 0.1147 | 0.2843 / 0.3045 | 0.5848 / 0.6288 | 0.6806 / 0.7624 |

Controls at the VALID-best tile strategy of the register (learned):

- ryd vs gutz: +0.06% (3/3, p=0.02)
- ryd vs ryd2d: +0.02% (3/3, p=0.17)
- ryd vs mlpX4: +0.01% (2/3, p=0.65)
- division of labor, hyb (register + MLP) vs mlp2 (two MLPs): -0.01% (1/3, p=0.66)

## RealPDE (U-Net) (trunk TEST 0.00948)

| strategy | U | lin gain / captured | mlp gain / captured | ryd gain / captured | ryd vs mlp |
|---|---|---|---|---|---|
| pooled | 0.0067 | +0.12% / 0.0022 (n=3) | +0.12% / 0.0023 (n=3) | +0.12% / 0.0022 (n=3) | -0.01% (1/3, p=0.25) |
| static | 0.0173 | +0.01% / 0.0004 (n=3) | +0.02% / 0.0004 (n=3) | +0.01% / 0.0004 (n=3) | -0.00% (0/3, p=0.07) |
| proxy | 0.0080 | -0.00% / -0.0000 (n=3) | -0.00% / -0.0000 (n=3) | -0.00% / -0.0000 (n=3) | -0.00% (0/3, p=0.40) |
| learned | 0.0162 | +0.01% / 0.0003 (n=3) | +0.01% / 0.0004 (n=3) | +0.01% / 0.0002 (n=3) | -0.00% (0/3, p=0.26) |
| gain | 0.0104 | +0.01% / 0.0003 (n=3) | +0.01% / 0.0004 (n=3) | +0.01% / 0.0004 (n=3) | -0.00% (0/3, p=0.03) |
| random | 0.0002 | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% (0/3, p=nan) |
| oracle (diagnostic) | 0.0543 | +0.03% / 0.0009 (n=1) | +0.04% / 0.0010 (n=1) | +0.03% / 0.0010 (n=1) | -0.00% (0/1, p=nan) |

K sweep (captured / U):

| strategy, core | K=10 | K=30 | K=100 | K=300 |
|---|---|---|---|---|
| pooled, mlp | 0.0023 / 0.0067 | 0.0023 / 0.0067 | 0.0022 / 0.0067 | 0.0025 / 0.0067 |
| pooled, ryd | 0.0022 / 0.0067 | 0.0022 / 0.0067 | 0.0020 / 0.0067 | 0.0024 / 0.0067 |
| proxy, mlp | 0.0000 / 0.0030 | -0.0000 / 0.0080 | 0.0000 / 0.0225 | 0.0014 / 0.0566 |
| proxy, ryd | 0.0000 / 0.0030 | -0.0000 / 0.0080 | 0.0000 / 0.0225 | 0.0018 / 0.0566 |

Controls at the VALID-best tile strategy of the register (static):

- ryd vs gutz: +0.00% (2/3, p=0.41)
- ryd vs ryd2d: +0.00% (2/3, p=0.43)
- ryd vs mlpX4: -0.00% (0/3, p=0.12)

## RealPDE (FNO) (trunk TEST 0.01413)

| strategy | U | lin gain / captured | mlp gain / captured | ryd gain / captured | ryd vs mlp |
|---|---|---|---|---|---|
| pooled | 0.0744 | +3.30% / 0.0717 (n=3) | +3.32% / 0.0721 (n=3) | +3.32% / 0.0721 (n=3) | -0.00% (2/3, p=0.71) |
| static | 0.0851 | +3.73% / 0.0806 (n=3) | +3.74% / 0.0808 (n=3) | +3.76% / 0.0812 (n=3) | +0.02% (3/3, p=0.00) |
| proxy | 0.0452 | +1.25% / 0.0309 (n=3) | +1.49% / 0.0360 (n=3) | +1.52% / 0.0365 (n=3) | +0.02% (3/3, p=0.13) |
| learned | 0.0913 | +3.81% / 0.0839 (n=3) | +3.83% / 0.0843 (n=3) | +3.84% / 0.0845 (n=3) | +0.01% (3/3, p=0.12) |
| gain | 0.0912 | +3.83% / 0.0844 (n=3) | +3.86% / 0.0849 (n=3) | +3.87% / 0.0850 (n=3) | +0.01% (3/3, p=0.18) |
| random | 0.0002 | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% / 0.0000 (n=3) | +0.00% (2/3, p=0.36) |
| oracle (diagnostic) | 0.1086 | +3.63% / 0.0802 (n=1) | +3.88% / 0.0850 (n=1) | +3.90% / 0.0853 (n=1) | +0.02% (1/1, p=nan) |

K sweep (captured / U):

| strategy, core | K=10 | K=30 | K=100 | K=300 |
|---|---|---|---|---|
| pooled, mlp | 0.0722 / 0.0744 | 0.0721 / 0.0744 | 0.0720 / 0.0744 | 0.0720 / 0.0744 |
| pooled, ryd | 0.0721 / 0.0744 | 0.0721 / 0.0744 | 0.0722 / 0.0744 | 0.0722 / 0.0744 |
| proxy, mlp | 0.0123 / 0.0161 | 0.0360 / 0.0452 | 0.0993 / 0.1236 | 0.1676 / 0.2273 |
| proxy, ryd | 0.0128 / 0.0161 | 0.0365 / 0.0452 | 0.0993 / 0.1236 | 0.1681 / 0.2273 |

Controls at the VALID-best tile strategy of the register (gain):

- ryd vs gutz: +0.04% (3/3, p=0.00)
- ryd vs ryd2d: +0.00% (1/3, p=0.89)
- ryd vs mlpX4: +0.00% (2/3, p=0.70)
- division of labor, hyb (register + MLP) vs mlp2 (two MLPs): +0.01% (2/3, p=0.25)

## pelzer (trunk TEST 0.11245)

| strategy | U | lin gain / captured | mlp gain / captured | ryd gain / captured | ryd vs mlp |
|---|---|---|---|---|---|
| pooled | 0.0471 | +1.32% / 0.0172 (n=3) | +1.38% / 0.0177 (n=3) | +1.45% / 0.0201 (n=3) | +0.08% (2/3, p=0.65) |
| static | 0.4588 | +8.43% / 0.1420 (n=3) | +9.39% / 0.1569 (n=3) | +9.23% / 0.1523 (n=3) | -0.18% (1/3, p=0.60) |
| proxy | 0.3209 | +2.54% / 0.0497 (n=3) | +3.33% / 0.0647 (n=3) | +2.71% / 0.0530 (n=3) | -0.64% (1/3, p=0.53) |
| learned | 0.4595 | +6.87% / 0.1262 (n=3) | +7.91% / 0.1433 (n=3) | +8.46% / 0.1518 (n=3) | +0.59% (2/3, p=0.63) |
| gain | 0.3498 | +6.65% / 0.1182 (n=3) | +7.01% / 0.1315 (n=3) | +7.24% / 0.1294 (n=3) | +0.23% (1/3, p=0.77) |
| random | 0.0116 | -0.01% / -0.0009 (n=3) | -0.01% / -0.0013 (n=3) | +0.01% / -0.0005 (n=3) | +0.02% (3/3, p=0.02) |
| oracle (diagnostic) | 0.6473 | +14.94% / 0.2548 (n=1) | +13.38% / 0.2287 (n=1) | +15.34% / 0.2613 (n=1) | +2.26% (1/1, p=nan) |

K sweep (captured / U):

| strategy, core | K=10 | K=30 | K=100 | K=300 |
|---|---|---|---|---|
| pooled, mlp | 0.0152 / 0.0471 | 0.0177 / 0.0471 | 0.0180 / 0.0471 | 0.0197 / 0.0471 |
| pooled, ryd | 0.0173 / 0.0471 | 0.0201 / 0.0471 | 0.0160 / 0.0471 | 0.0205 / 0.0471 |
| proxy, mlp | 0.0201 / 0.1884 | 0.0647 / 0.3209 | 0.1126 / 0.5364 | 0.1563 / 0.7195 |
| proxy, ryd | 0.0333 / 0.1884 | 0.0530 / 0.3209 | 0.1399 / 0.5364 | 0.1209 / 0.7195 |

Controls at the VALID-best tile strategy of the register (static):

- ryd vs gutz: +0.40% (1/3, p=0.78)
- ryd vs ryd2d: -0.74% (0/3, p=0.02)
- ryd vs mlpX4: -0.18% (1/3, p=0.79)
- division of labor, hyb (register + MLP) vs mlp2 (two MLPs): -0.15% (2/3, p=0.57)

## MHD_64 (trunk TEST 0.54470)

| strategy | U | lin gain / captured | mlp gain / captured | ryd gain / captured | ryd vs mlp |
|---|---|---|---|---|---|
| pooled | 0.0020 | +0.11% / 0.0020 (n=3) | +0.11% / 0.0020 (n=3) | +0.11% / 0.0020 (n=3) | +0.00% (3/3, p=0.09) |
| static | 0.0088 | +0.36% / 0.0082 (n=3) | +0.36% / 0.0082 (n=3) | +0.36% / 0.0082 (n=3) | -0.00% (1/3, p=0.30) |
| proxy | 0.0206 | +0.77% / 0.0195 (n=3) | +0.77% / 0.0195 (n=3) | +0.77% / 0.0195 (n=3) | +0.00% (2/3, p=0.24) |
| learned | 0.0229 | +0.87% / 0.0219 (n=3) | +0.87% / 0.0219 (n=3) | +0.87% / 0.0219 (n=3) | +0.00% (2/3, p=0.58) |
| gain | 0.0229 | +0.87% / 0.0219 (n=3) | +0.87% / 0.0218 (n=3) | +0.87% / 0.0219 (n=3) | +0.00% (1/3, p=0.58) |
| random | 0.0001 | +0.00% / 0.0001 (n=3) | +0.00% / 0.0001 (n=3) | +0.00% / 0.0001 (n=3) | -0.00% (0/3, p=0.03) |
| oracle (diagnostic) | 0.0251 | +0.94% / 0.0236 (n=1) | +0.94% / 0.0236 (n=1) | +0.94% / 0.0236 (n=1) | +0.00% (1/1, p=nan) |

K sweep (captured / U):

| strategy, core | K=10 | K=30 | K=100 | K=300 |
|---|---|---|---|---|
| pooled, mlp | 0.0020 / 0.0020 | 0.0020 / 0.0020 | 0.0020 / 0.0020 | 0.0020 / 0.0020 |
| pooled, ryd | 0.0020 / 0.0020 | 0.0020 / 0.0020 | 0.0020 / 0.0020 | 0.0020 / 0.0020 |
| proxy, mlp | 0.0089 / 0.0094 | 0.0195 / 0.0206 | 0.0422 / 0.0451 | 0.0790 / 0.0864 |
| proxy, ryd | 0.0089 / 0.0094 | 0.0195 / 0.0206 | 0.0422 / 0.0451 | 0.0791 / 0.0864 |

Controls at the VALID-best tile strategy of the register (gain):

- ryd vs gutz: -0.00% (0/3, p=0.21)
- ryd vs ryd2d: -0.00% (0/3, p=0.10)
- ryd vs mlpX4: -0.00% (0/3, p=0.17)
- division of labor, hyb (register + MLP) vs mlp2 (two MLPs): -0.00% (2/3, p=0.75)

# H_U: utilization governs the gain

- lin: Spearman(U, captured) over 42 (bench x strategy) cells = 0.70 [bench-bootstrap 95% CI 0.10, 0.98]
- mlp: Spearman(U, captured) over 42 (bench x strategy) cells = 0.71 [bench-bootstrap 95% CI 0.17, 0.98]
- ryd: Spearman(U, captured) over 42 (bench x strategy) cells = 0.71 [bench-bootstrap 95% CI 0.18, 0.98]
- mlp: causal routing <= pooled on 5/7 benchmarks (P2); gain routing <= energy routing on 3/7 (P4)
- ryd: causal routing <= pooled on 5/7 benchmarks (P2); gain routing <= energy routing on 3/7 (P4)

# H_Q: register vs matched MLP at each benchmark's VALID-best tile strategy

- ryd vs mlp: better/worse 3/3 (sign p=1.00); random-effects +0.006% [-0.005, +0.017], I2=98%
  - TRL2D (static): +0.000% (0/3)
  - rayleigh_benard (proxy): -0.479% (1/3)
  - shear_flow (learned): +0.026% (3/3)
  - RealPDE (U-Net) (static): -0.002% (0/3)
  - RealPDE (FNO) (gain): +0.006% (3/3)
  - pelzer (static): -0.181% (1/3)
  - MHD_64 (gain): +0.001% (1/3)
- ryd vs lin: better/worse 6/1 (sign p=0.12); random-effects +0.026% [+0.005, +0.047], I2=99%
  - TRL2D (static): +0.036% (2/3)
  - rayleigh_benard (proxy): +10.524% (3/3)
  - shear_flow (learned): +0.041% (3/3)
  - RealPDE (U-Net) (static): +0.000% (2/3)
  - RealPDE (FNO) (gain): +0.036% (3/3)
  - pelzer (static): +0.871% (3/3)
  - MHD_64 (gain): -0.001% (1/3)
