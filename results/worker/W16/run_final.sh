#!/bin/bash
ROOT=$1; CONCS=$2; mkdir -p $ROOT
for S in 1 2 3; do for ARM in a b; do
  [ $S = 3 ] && export GSM_N=2 || export GSM_N=3
  /scratch/kopt-w16/start.sh $HOME/vllm-kopt-w16$ARM $ROOT/${ARM}$S "$CONCS" /scratch/kopt-w16/after2.sh
  echo "${ARM}$S $(tail -1 $ROOT/${ARM}$S/log) $(ls $ROOT/${ARM}$S | grep -c FAILED) failed" >> $ROOT/status
done; done
echo ALLDONE >> $ROOT/status
