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



HIP side data  inference 



