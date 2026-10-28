#!/bin/bash
# neuraloperator 0.3.0 + torch >= 2.9: SpectralConv indexes Fourier modes with a Python LIST of slices
# (spectral_convolution.py:453/459); torch 2.9+ interprets a list as a tensor index -> silently wrong outputs
# (official TFNO checkpoint gave VRMSE 15 instead of ~0.1 on planetswe). Make the indices tuples (the rest of the file already does).
F=/home/mohhakami/Documents/paper/quworld/.venv-well/lib/python3.13/site-packages/neuralop/layers/spectral_convolution.py
sed -i 's/weight = self._get_weight(indices)\[slices_w\]/weight = self._get_weight(indices)[tuple(slices_w)]/; s/out_fft\[slices_x\] = self._contract(x\[slices_x\], weight, separable=False)/out_fft[tuple(slices_x)] = self._contract(x[tuple(slices_x)], weight, separable=False)/' "$F"
grep -n "tuple(slices_w)\|tuple(slices_x)" "$F"
