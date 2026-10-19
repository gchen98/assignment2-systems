#!/bin/bash

config_dir='configs/'
scripts_dir='modal_benchmarking/'
label='small'
model_params_file="${config_dir}/random_config_${label}.yaml"

uv run modal run ${scripts_dir}trainer.py --kwargs-json-path ${config_dir}profiler.json --model-params-file ${model_params_file} --function transformer_iteration  --with-stack --label ${label}
exit 0
