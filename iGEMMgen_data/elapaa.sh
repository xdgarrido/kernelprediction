#!/bin/bash
export ROCR_VISIBLE_DEVICES=$GPU
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_HSACO=out/igemm_bwd_gtc_gfx908_nhwc.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_PRINT_ALL_GKS=1
export IGEMM_GKS_ITERATIVE=1

