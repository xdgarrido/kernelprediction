#!/bin/bash
export ROCR_VISIBLE_DEVICES=0
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_HSACO=out/igemm_fwd_gtc_gfx908_nhwc.hsaco
export IGEMM_TENSOR_CAST_HSACO=./out/igemm_gtc_tensor_cast.hsaco
export IGEMM_CONFIG_FILE=./config/igemm_fwd_gtc_gfx908_nhwc.config
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_PRINT_ALL_GKS=1
export IGEMM_GKS_ITERATIVE=1
export IGEMM_MODE=1


export ROCR_VISIBLE_DEVICES=0
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_HSACO=out/igemm_bwd_gtc_gfx908_nhwc.hsaco
export IGEMM_TENSOR_CAST_HSACO=./out/igemm_gtc_tensor_cast.hsaco
export IGEMM_CONFIG_FILE=./config/igemm_bwd_gtc_gfx908_nhwc.config
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_PRINT_ALL_GKS=1
export IGEMM_GKS_ITERATIVE=1
export IGEMM_MODE=0

export ROCR_VISIBLE_DEVICES=0
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_HSACO=out/igemm_wrw_gtc_gfx908_nhwc.hsaco
export IGEMM_TENSOR_CAST_HSACO=./out/igemm_gtc_tensor_cast.hsaco
export IGEMM_CONFIG_FILE=./config/igemm_wrw_gtc_gfx908_nhwc.config
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_PRINT_ALL_GKS=1
export IGEMM_GKS_ITERATIVE=1
export IGEMM_MODE=0

# export IGEMM_MAX_MPB=64
# export IGEMM_MAX_NPB=64
# export IGEMM_MAX_GKS=8
./out/conv_driver.exe conv  -n 384 -c 1824 -H 32 -W 32 -k 224 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 1 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC >> pred_results0.txt

