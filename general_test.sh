#!/bin/bash

data_path=".."
dataset="Fluo-C2DL-Huh7"
seq="01"
mode="GT"
model_path="models/Fluo-C2DL-Huh7-${mode}.t7"

jcell segment --input=${data_path}/${dataset}/${seq}/ --model=${model_path} --output=../../${dataset}/${seq}_RES-${mode}/ --use_gpu=1 --overwrite --output_type instances --AAF