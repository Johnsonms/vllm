#!/bin/bash
# group ablation: rounds of [base, a, b, c, d, e]; usage run_abl.sh ROOT "CONCS" NROUNDS
ROOT=$1; CONCS=$2; N=${3:-2}; mkdir -p $ROOT
for S in $(seq 1 $N); do for G in base a b c d e; do
  if [ $G = base ]; then /scratch/kopt-w16/start.sh $HOME/vllm-kopt-w16a $ROOT/${G}$S "$CONCS"
  else KOPT_W16_GROUPS=$G /scratch/kopt-w16/start.sh $HOME/vllm-kopt-w16c $ROOT/${G}$S "$CONCS"; fi
  echo "${G}$S $(tail -1 $ROOT/${G}$S/log) $(ls $ROOT/${G}$S | grep -c FAILED) failed" >> $ROOT/status
done; done
echo ALLDONE >> $ROOT/status
