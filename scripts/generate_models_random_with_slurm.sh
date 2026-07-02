#!/bin/bash

# Generate abstract-input corpora using the random sampling strategy
# (generator.z3_random) across all API variants in parallel via Slurm.
#
# Usage:
#   bash scripts/generate_models_random_with_slurm.sh [duration] [n_max] [lib] [seed] [regen]
#
# Outputs are written to:
#   corpus_torch_random/<variant>/abstract_inputs.jsonl  (for lib=torch)
#   corpus_tf_random/<variant>/abstract_inputs.jsonl     (for lib=tf)
#
# A per-variant summary CSV is aggregated at:
#   .tmp/model_generation_random_<lib>.csv

duration=${1:-300}    # wall-clock budget per variant in seconds
n_max=${2:-0}         # max valid models to generate (0 = unlimited)
lib=${3:-torch}       # library: torch or tf
seed=${4:-200}        # RNG seed
regen=${5:-0}         # 1 = overwrite existing corpus

# Normalise library alias
if [ "$lib" = "pytorch" ]; then
  lib=torch
elif [ "$lib" = "tensorflow" ]; then
  lib=tf
fi

# Default to the standard variations file for the given library
if [ -z "${elements_file}" ]; then
  export elements_file=${lib}_variations.txt
fi

export OMP_NUM_THREADS=1           # prevent coverage-collection threading issues
export TF_ENABLE_ONEDNN_OPTS=0    # disable oneDNN optimisations for TensorFlow

# Add 1 h buffer to the Slurm time limit to account for rule loading
total_seconds=$((duration + 3600))
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))
seconds=$((total_seconds % 60))
export slurm_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

job_name=modlr
slurm_sh=$(dirname "$(realpath "$0")")/slurm_base.sh

bash $slurm_sh "python -m generator.z3_random" ${job_name} ${duration} ${n_max} ${lib} ${seed} ${regen}

PROJECT_DIR=$(dirname "$(realpath "$0")")/../

# Aggregate per-variant CSV files written by z3_random into a single summary
tmp_results=$PROJECT_DIR/.tmp/model_results
result=$PROJECT_DIR/.tmp/model_generation_random_${lib}.csv
echo "api,unsat,nominal,invalid,crash,exception,total,valid_prcnt" > "${result}"
for filename in ${tmp_results}/*.csv; do
    cat "${filename}" >> "${result}"
done

echo "Random model-gen results saved in ${result}"
