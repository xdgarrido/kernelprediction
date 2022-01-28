#!/bin/bash
export ROCR_VISIBLE_DEVICES=$GPU
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_HSACO=out/igemm_wrw_gtc_gfx908_nhwc.hsaco
export IGEMM_TENSOR_CAST_HSACO=out/igemm_gtc_tensor_cast.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_GKS_ITERATIVE=1
export IGEMM_MAX_MPB=64
export IGEMM_MAX_NPB=64
export IGEMM_MAX_GKS=8
DECK=13
count=0
# 1 by 1 convolutions
X=1
Y=1
# Strides 
U=1
V=1
# Dilation
J=1 
L=1 
# Padding 
P=0
Q=0 
# Groups 
G=1

for BATCH in 4 8 16 32 48 64 96 128 192 256 288 320 384 512 768 1024
  do 
    for WIDTH_HEIGHT in 4 6 7 8 10 13 14 15 17 16 20 23 24 30 32 35 36 39 42 45 48 52 56 58 62 64 68 72 84 96 112 128
    do 
      for IC in 32 64 96 128 224 256 320 384 512 768 1024 1536 1824 2048 3072 4096
        do 
           for OC in  2048
           do   
              # run conv_driver
               nohup echo "$BATCH : $WIDTH_HEIGHT : $IC : $OC " >> conv_script$DECK.txt 2>&1 &
               ./out/conv_driver.exe conv -V 0 -i 5 -n $BATCH -c $IC -H $WIDTH_HEIGHT -W $WIDTH_HEIGHT -k $OC -y 1 -x 1 -p $P -q $Q -u $U -v $V -l $L -j $J -g $G -F 4 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -B ./conv_data$DECK.txt 2>&1 | tee >> conv_output$DECK.txt
               let "count++"
           done 
        done 
    done 
done
