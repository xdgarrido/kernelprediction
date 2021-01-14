Instructions for usage

There are 3 sub-directories:-

1. filtercluster : performs design of multiple classifiers

Typical example:
./filtercluster -i conv_data.txt -g quant -n 10000:8000:6000:4000 -m 1 -p 2000

it generates files labels.csv (kernel parameters set) ts.csv (training set) cs.csv (test set) quant*.txt (classifiers)
cp quant6000.txt labels.csv cs.csv domain.csv ../predictparams/

2. predictparams : performs classification depending on the number of allowed predictors
Copy the files above to ./predictparams (results of classification outside of ts)

./predictparams -i cs.csv -g quant6000.txt -n 20 -c 1
./predictparams -i cs.csv -g quant6000.txt -n 10 -c 1

3. learncluster : iterates previously designed classifiers 

Improvements related with GLVQ (generalized learning vector quantization)
It is necessary the following files in the learncluster directory

cp cs.csv ts.csv quant6000.txt domain.csv ../learncluster/
./learncluster -g quant6000.txt -n 10 -c 10000 -r 2000 -l 0.00005 -e 3

Now we can check if there was improvement outside ts

cp quant6000_opt.txt ../predictparams/
./predictparams -i cs.csv -g quant6000_opt.txt -n 10 -c 1


