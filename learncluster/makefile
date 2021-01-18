CC=g++
CFLAGS=-I. -O2 -D__AVXACC__ -DLINUX -march=native -mavx
LIBS= -lm

default: learncluster

learning.o: learning.cpp learning.h
	$(CC) $(CFLAGS) -c learning.cpp

parse.o: parse.cpp parse.h 
	$(CC) $(CFLAGS) -c parse.cpp

main.o: main.cpp parse.h learning.h
	$(CC) $(CFLAGS) -c main.cpp

learncluster: main.o parse.o learning.o
	$(CC) $(CFLAGS) $(LIBS) -o learncluster main.o parse.o learning.o

clean: 
	rm -rf *.o
real_clean: 
	rm -rf *.o *.csv *.txt 	
