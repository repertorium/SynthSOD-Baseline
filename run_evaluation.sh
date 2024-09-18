#!/bin/bash

# Modified from the MUSDB18 example of the asteroid library.

# Exit on error
set -e
set -o pipefail

python_path=python

# Example usage
# ./run_evaluation.sh --tag ensembleset

# General
tag=synthsod  # Controls the directory name associated to the experiment
eval_on="synthsod_test" # Controls the dataset used for evaluation. Options: synthsod_test, synthsod_train, ensembleset, aalto, urmp
                        # You can use several separating them by spaces (e.g. "aalto synthsod_test")

. parse_options.sh

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi
expdir=exp/train_xumx_${tag}
mkdir -p ${expdir}_eval && echo $uuid >> ${expdir}_eval/run_uuid.txt
echo "Results from the following experiment will be stored in ${expdir}_eval"

$python_path eval.py \
  --no-cuda \
  --models ${expdir}_strings/best_model.pth ${expdir}_woodwinds/best_model.pth ${expdir}_brass/best_model.pth ${expdir}_percussion/best_model.pth \
  --eval_on $eval_on \
  --outdir ${expdir}_eval | tee logs/eval_${tag}.log
cp logs/eval_${tag}.log ${expdir}_eval/eval.log
