# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
NUM_GPU=8
TRAIN_CFG=/shared/deep-learning-models/models/vision/detection/configs/faster_rcnn/EC2/faster_rcnn_r50_fpn_1x_coco.py

echo ""
echo "NUM_GPU: ${NUM_GPU}"
echo "TRAIN_CFG: ${TRAIN_CFG}"
echo ""

cd /shared/deep-learning-models/models/vision/detection
export PYTHONPATH=${PYTHONPATH}:${PWD}

/opt/amazon/openmpi/bin/mpirun \
--allow-run-as-root \
--mca plm_rsh_no_tree_spawn 1 \
--tag-output \
--hostfile /shared/hosts_1 \
-N 8 \
-mca btl_tcp_if_exclude lo,docker0 \
-x LD_LIBRARY_PATH \
-x PATH \
-x FI_PROVIDER="efa" \
-x NCCL_DEBUG=INFO \
-x TF_CUDNN_USE_AUTOTUNE=0 \
-x PYTHONPATH \
--oversubscribe \
bash /shared/deep-learning-models/models/vision/detection/scripts/launcher.sh \
python tools/train.py \
--config ${TRAIN_CFG} \
--autoscale-lr \
--amp
