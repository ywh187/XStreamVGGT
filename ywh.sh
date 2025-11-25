source /mnt/bn/automl-aigc/weihaoye/miniconda3/bin/activate && conda activate StreamVGGT



export http_proxy=http://bj-rd-proxy.byted.org:3128  https_proxy=http://bj-rd-proxy.byted.org:3128 no_proxy=code.byted.org 


cd /mnt/bn/automl-aigc/weihaoye/StreamVGGT
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 \
accelerate launch --multi_gpu --main_process_port $ARNOLD_WORKER_0_PORT ./src/train.py --config-name train




# [2025-11-24 18:47:26,958][croco.utils.misc][INFO] - Epoch: [0]  [2400/4500]  eta: 1:21:08  lr: 0.000010  epoch: 0.5311 (0.2667)  step: 2390.0000 (1199.9633)  loss: -0.4813 (-0.5008)  Lcamera: 0.9408 (1.2004)  Ldepth: -0.9729 (-1.2830)  Lpmap: -0.1523 (-0.4303)  Ltrack: 0.0000 (0.0121)  total: -0.4813 (-0.5008)  time: 2.0950  data: 0.1006  max mem: 67786



