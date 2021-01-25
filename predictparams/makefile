CC=g++
CFLAGS=-I. -O2 -D__AVXACC__ -DLINUX -march=native -mavx
LIBS= -lm

default: predictparams

classifier.o: classifier.cpp classifier.h 
	$(CC) $(CFLAGS) -c classifier.cpp

parse.o: parse.cpp parse.h 
	$(CC) $(CFLAGS) -c parse.cpp

main.o: main.cpp parse.h classifier.h
	$(CC) $(CFLAGS) -c main.cpp

predictparams: main.o parse.o classifier.o
	$(CC) $(CFLAGS) $(LIBS) -o predictparams main.o parse.o classifier.o

clean: 
	rm -rf *.o
real_clean: 
	rm -rf *.o *.txt *.csv 
