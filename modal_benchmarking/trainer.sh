#!/bin/bash

config_dir='configs/'
scripts_dir='modal_benchmarking/'
label='large'
model_params_file="${config_dir}/random_config_${label}.yaml"
timing_flag='--timing'
memory_flag='--profile-memory'


uv run modal run ${scripts_dir}trainer.py --kwargs-json-path ${config_dir}profiler.json --model-params-file ${model_params_file} --function transformer_iteration  ${timing_flag} ${memory_flag} --with-stack --label ${label}
exit 0
