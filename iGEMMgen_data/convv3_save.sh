#!/bin/bash
export ROCR_VISIBLE_DEVICES=$GPU
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_HSACO=out/igemm_wrw_gtc_gfx908_nhwc.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_GKS_ITERATIVE=0
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

for BATCH in 1 2 4 8 12 16
  do 
    for WIDTH in 176 256 352 426 512 640 854 1280 1920 2560 3840 
    do 
      for HEIGHT in 144 240 288 256 360 480 512 720 1080 1440 2160
      do
      for IC in 1 2 4 8 12 16
        do 
           for OC in 16
           do   
              # run conv_driver
               nohup echo "$BATCH : $WIDTH_HEIGHT : $IC : $OC " >> convv_script$ROCR_VISIBLE_DEVICES.txt 2>&1 &
               ./out/conv_driver.exe conv -V 0 -i 5 -n $BATCH -c $IC -H $HEIGHT -W $WIDTH -k $OC -y 1 -x 1 -p $P -q $Q -u $U -v $V -l $L -j $J -g $G -F 4 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -B ./convv_data$ROCR_VISIBLE_DEVICES.txt 2>&1 | tee >> convv_output$ROCR_VISIBLE_DEVICES.txt
               let "count++"
           done 
          done
        done 
    done 
done
