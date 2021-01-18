CC=g++
CFLAGS=-I. -O2 -D__AVXACC__ -DLINUX -march=native -mavx
LIBS= -lm

default: filtercluster

clustering.o: clustering.cpp clustering.h
	$(CC) $(CFLAGS) -c clustering.cpp

preprocessing.o: preprocessing.cpp preprocessing.h
	$(CC) $(CFLAGS) -c preprocessing.cpp

parse.o: parse.cpp parse.h 
	$(CC) $(CFLAGS) -c parse.cpp

main.o: main.cpp parse.h clustering.h preprocessing.h
	$(CC) $(CFLAGS) -c main.cpp

filtercluster: main.o parse.o clustering.o preprocessing.o
	$(CC) $(CFLAGS) $(LIBS) -o filtercluster main.o parse.o clustering.o preprocessing.o

clean: 
	rm -rf *.o
real_clean: 
	rm -rf *.o *.csv hist.txt quant*.txt
