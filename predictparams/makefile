CC=g++
CFLAGS=-I. -O2 -D__AVXACC__ -DLINUX -march=native -mavx
LIBS= -lm

default: predictparams

parse.o: parse.cpp parse.h 
	$(CC) $(CFLAGS) -c parse.cpp

main.o: main.cpp parse.h
	$(CC) $(CFLAGS) -c main.cpp

predictparams: main.o parse.o
	$(CC) $(CFLAGS) $(LIBS) -o predictparams main.o parse.o

clean: 
	rm -rf *.o
real_clean: 
	rm -rf *.o *.txt *.csv 
