#!/bin/bash

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
PYTHON=python

INFERENCE_CODE=inference.py

dataset= 'dfc2019'
exp_name= 'pointtransformer_repro'
data=$3
exp_dir='/local/scratch0/hanyang/Codes/point_cloud_visualizer/tmp_dir'
tile_dir='tiles'
model_dir='/local/scratch0/hanyang/Codes/point_cloud_visualizer/logs/point_transformer'
result_dir='/local/scratch0/hanyang/Codes/point_cloud_visualizer/tmp_dir/tile_classification_results'
config='/local/scratch0/hanyang/Codes/point_cloud_visualizer/point_transformer/dfc2019_pointtransformer_repro.yaml'
mkdir -p ${tile_dir}
mkdir -p ${result_dir}

now=$(date +"%Y%m%d_%H%M%S")
cp ${config} tool/inference.sh tool/${INFERENCE_CODE} ${exp_dir}

$PYTHON scripts/process_tiling.py --in_path ${data} --out_dir ${tile_dir}

$PYTHON -u ${exp_dir}/${INFERENCE_CODE} \
  --config=${config} \
  --data_dir ${tile_dir} \
  save_folder ${result_dir} \
  model_path ${model_dir}/model_best.pth \
  2>&1 | tee ${result_dir}/test_best-$now.log

$PYTHON scripts/stitching_tiles.py --in_pkl_dir ${tile_dir} \
  --in_label_dir ${result_dir} \
  --in_label_suffix pred \
  --out_las_path ${result_dir}/pred.las

$PYTHON scripts/resample_classification.py ${data} ${result_dir}/pred.las ${result_dir}/pred_resampled.laz
$PYTHON scripts/remap_classification.py ${result_dir}/pred_resampled.laz ${result_dir}/pred_resampled_USGS.laz
