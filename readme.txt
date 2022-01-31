Instructions for usage

There are 9 sub-directories:-

1. filtercluster : performs design of multiple classifiers

/filtercluster
Options
  -i input file name
  -j trainning set file name
  -g quantizer file name
  -k verification set file name
  -v verbose
  -n number of clusters 
  -f 0: naive pnn 1: fast pnn 
  -m 0: do not apply normalization 1: apply min-max normalization 2: apply z-score normalization
  -p number of clusters used for the cs set
  -s convolution kernel size: (1) 1by1 [default] or (n) nbyn

Typical example:
./filtercluster -i conv.txt -g quant -n 10000:8000:6000:4000 -m 1 -p 2000 -s 1
[conv.txt is the data found by the data aquisition. It can be found in the mi1001by1 directory].

it generates files labels.csv (asm kernel parameters set), ts.csv (training set), cs.csv (test set), cs_norm.csv (normalized test set),
quant*.txt (classifiers), domain.csv (scales used in the normalization process) hist.txt (histogram of clusters mapped into label.set) 

2. These classifiers now can be improved by using GMLVQ algorithm. This is done using the matlab code in ./lvq_mat. The files 
cs.csv, cs_norm.csv, domain.csv, hist8000.txt, labels.csv, quant8000.csv and ts.csv are designed by filtercluster and initializes 
the learning phase. Run the matlab script: gmlvq_bench.m 

Check gmlvq_bench.m to see the options. After 100 epochs, the optimized classifier will be written into the files:
quant8000_opt.csv and omega.csv (weighting matrix).  

cp quant8000_opt.txt omega.csv labels.csv cs.csv domain.csv ../predictparams/

3. predictparams : performs classification depending on the number of allowed predictors
Copy the files above to ./predictparams (results of classification outside of ts)

./predictparams
Options:
  -i convolution parameters file name (ex: cs.csv)
  -g quantizer file name (ex: quant6000.csv)
  -l labels file name (ex: labels.csv)
  -m normaization scales file name (ex: domain.csv)
  -c 0: do not apply normalization 1: apply min-max normalization 2: apply z-score normalization
  -n number of predictors
  -v verbose
  -s classifier used (ex: none (euclidian distance, lambdas.csv (weighted euclidian distance), omega.csv (mahalanobis distance) 
  -k kernel type :1 (1by1) and n(nbyn)
  -t conv_type: (fwd,bwd,wrw)
  -p precision: fp32 (default) or  fp16
  -f layout: nhwc (default) or nchw

./predictparams -i cs.csv -g quant8000_opt.csv -l labels.csv  -m domain.csv -n 1 -c 1 -s omega.csv -k 1 -t fwd -p fp32 -f nhwc 
It is used as accuracy checker. 


4. gensh is used to fill the gaps on the domain of convolutions in the data collection step. 

5. inference is used to develop routines to hook up into the hip conv_driver

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

HIP side data collection (iGEMMgen_data)

In general hooks have to be added into the driver for saving convolution data two files have to be changed:
conv_driver.cpp and args.h. The shell scripts conv[0-15].sh runs the conv_driver in multiple GPU's. Please 
check them to understand env. variables. Compilation is done using the following commands:

fwd fp32 : python3 igemm_codegen.py config/igemm_fwd_gtc_gfx908_nhwc.config
bwd fp32 : python3 igemm_codegen.py config/igemm_bwd_gtc_gfx908_nhwc.config
wrw fp32 : python3 igemm_codegen.py config/igemm_wrw_gtc_gfx908_nhwc.config



HIP side data  inference (iGEMMgen_inference)

call drun.sh first
compile code with python script 
fwd fp32 : python3 igemm_codegen.py config/igemm_fwd_gtc_gfx908_nhwc.config

bwd fp32 : python3 igemm_codegen.py config/igemm_bwd_gtc_gfx908_nhwc.config

wrw fp32 : python3 igemm_codegen.py config/igemm_wrw_gtc_gfx908_nhwc.config

set env. variables for execution
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
export IGEMM_MODE=0 (1 uses classification)


export ROCR_VISIBLE_DEVICES=0
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_HSACO=out/igemm_bwd_gtc_gfx908_nhwc.hsaco
export IGEMM_TENSOR_CAST_HSACO=./out/igemm_gtc_tensor_cast.hsaco
export IGEMM_CONFIG_FILE=./config/igemm_bwd_gtc_gfx908_nhwc.config
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_PRINT_ALL_GKS=1
export IGEMM_GKS_ITERATIVE=1
export IGEMM_MODE=0 (1 uses classification)

export ROCR_VISIBLE_DEVICES=0
export IGEMM_GPU_NAIVE_CONV_HSACO=out/naive_conv.hsaco
export IGEMM_HSACO=out/igemm_wrw_gtc_gfx908_nhwc.hsaco
export IGEMM_TENSOR_CAST_HSACO=./out/igemm_gtc_tensor_cast.hsaco
export IGEMM_CONFIG_FILE=./config/igemm_wrw_gtc_gfx908_nhwc.config
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=1
export IGEMM_PRINT_ALL_GKS=1
export IGEMM_GKS_ITERATIVE=1
export IGEMM_MODE=0 (1 uses classification)

execute convolutions

# bwd convolutions
./out/conv_driver.exe conv  -n 96 -c 768 -H 48 -W 48 -k 384 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 2 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 96 -c 1824 -H 36 -W 36 -k 224 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 2 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 4 -c 320 -H 32 -W 32 -k 768 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 2 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 768 -c 224 -H 23 -W 23 -k 512 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 2 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 48 -c 128 -H 128 -W 128 -k 128 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 2 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
# fwd convolutions
./out/conv_driver.exe conv  -n 16 -c 2 -H 512 -W 1920 -k 12 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 1 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 384 -c 1824 -H 32 -W 32 -k 224 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 1 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 32 -c 1536 -H 56 -W 56 -k 224 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 1 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 12 -c 1 -H 360 -W 352 -k 1 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 1 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 192 -c 1536 -H 15 -W 15 -k 3072 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 1 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
# wrw convolutions
./out/conv_driver.exe conv  -n 48 -c 4096 -H 7 -W 7 -k 32 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 4 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 768 -c 512 -H 52 -W 52 -k 320 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 4 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 4 -c 4 -H 360 -W 426 -k 16 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 4 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 2 -c 2 -H 288 -W 854 -k 16 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 4 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 
./out/conv_driver.exe conv  -n 96 -c 2048 -H 56 -W 56 -k 2048 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -p 0 -q 0 -g 1 -F 4 -V 0 -i 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC 



