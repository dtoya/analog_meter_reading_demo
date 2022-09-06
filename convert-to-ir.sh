#!/bin/bash
data_root=./result
mo --input_model $data_root/weight_final.onnx --data_type FP16 --model_name analog-meter --output_dir $data_root/FP16 --scale_values [128,128,128] --mean_values [128,128,128]


