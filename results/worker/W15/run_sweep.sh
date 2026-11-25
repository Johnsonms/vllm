#!/bin/bash
# usage: run_sweep.sh GPU M [M...]
source ~/kopt-env.sh; cd ~/vllm-kopt-w14; GPU=$1; shift
for M in "$@"; do PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPU python /scratch/kopt-w11/sweep_skinny_generic.py /scratch/kopt-w15/shapes_m$M.json /scratch/kopt-w15/sweep/m$M.jsonl $M > /scratch/kopt-w15/sweep/m$M.log 2>&1; echo "M=$M done $(date -u +%H:%MZ)" >> /scratch/kopt-w15/sweep/status; done
