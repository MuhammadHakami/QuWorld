| row | strategy | captured | predictability failure | allocation-rule failure | budget failure (beyond K=30) | worst 10% windows: share of metric / gain | rest gain | early-time gain |
|---|---|---|---|---|---|---|---|---|
| turbulent_radiative_layer_2D | static | 0.000 | 0.231 | 0.229 | 0.540 | 0.18 / +0.0% | +0.0% | +0.0% |
| rayleigh_benard | proxy | 0.645 | 0.010 | 0.000 | 0.345 | 0.90 / +39.2% | +16.0% | +38.6% |
| shear_flow | learned | 0.495 | 0.000 | 0.001 | 0.503 | 0.68 / +26.5% | +26.0% | +20.8% |
| MHD_64 | gain | 0.025 | 0.001 | 0.002 | 0.972 | 0.13 / +1.1% | +0.8% | +0.8% |
| realpde_unet | static | 0.000 | 0.021 | 0.043 | 0.935 | 0.17 / -0.0% | +0.0% |  |
| realpde_fno | gain | 0.065 | 0.008 | 0.025 | 0.901 | 0.15 / +0.7% | +4.4% |  |
| pelzer | static | 0.188 | 0.239 | 0.170 | 0.404 | 0.18 / +11.5% | +8.6% |  |

By field (share of the metric-weighted error before; fraction removed):

- turbulent_radiative_layer_2D: density 0.02 (+0.0%), pressure 0.71 (+0.0%), vx 0.12 (+0.0%), vy 0.14 (+0.0%)
- rayleigh_benard: buoyancy 0.00 (-0.2%), pressure 0.00 (+0.7%), vx 0.42 (+66.0%), vy 0.58 (+63.5%)
- shear_flow: tracer 0.00 (+3.5%), pressure 0.97 (+50.6%), vx 0.00 (+3.2%), vy 0.03 (+8.9%)
- MHD_64: density 0.53 (+3.4%), Bx 0.13 (+1.5%), By 0.15 (+2.5%), Bz 0.09 (+0.8%), vx 0.03 (+0.4%), vy 0.03 (+0.5%), vz 0.03 (+0.6%)
- realpde_unet: u 0.23 (+0.1%), v 0.77 (-0.0%)
- realpde_fno: u 0.33 (+18.8%), v 0.67 (+0.4%)
- pelzer: temperature 1.00 (+18.8%)
