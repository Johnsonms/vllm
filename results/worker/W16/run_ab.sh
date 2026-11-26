#!/bin/bash
# interleaved restarts A B A B A B; usage run_ab.sh ROOT "CONCS" NSTARTS
ROOT=$1; CONCS=$2; N=${3:-3}; mkdir -p $ROOT
for S in $(seq 1 $N); do for ARM in a b; do
  /scratch/kopt-w16/start.sh $HOME/vllm-kopt-w16$ARM $ROOT/${ARM}$S "$CONCS" /scratch/kopt-w16/after.sh
  echo "${ARM}$S $(tail -1 $ROOT/${ARM}$S/log) $(ls $ROOT/${ARM}$S | grep -c FAILED) failed" >> $ROOT/status
done; done
echo ALLDONE >> $ROOT/status
