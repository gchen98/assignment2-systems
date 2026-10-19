#!/bin/bash

size_config='large'
model_params_file="data/random_config_${size_config}.yaml"

uv run modal run trainer.py --kwargs-json-path profiler.json --model-params-file ${model_params_file} --function transformer_iteration
exit 0
