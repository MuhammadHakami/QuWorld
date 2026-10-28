#!/bin/bash
# 21:06: memguard killed the F3 public build twice (F3 + BC VT + QHAR2 s3 > memory). Serial chain: F3 build + upload, then the §42c
# read-out (BC), then §42h (PQ2) and its conditional promotion. Each runner skips finished work.
cd /home/mohhakami/Documents/paper/quworld
bash runs/s42_f3.sh
bash runs/s42bc.sh
bash runs/s42h.sh
bash runs/s42h_promote.sh
