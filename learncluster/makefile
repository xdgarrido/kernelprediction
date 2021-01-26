CC=g++
CFLAGS=-I. -D__AVXACC__ -DLINUX -march=native -mavx
LIBS= -lm

default: learncluster

learning.o: learning.cpp learning.h
	$(CC) $(DEBUG) $(RELEASE) $(CFLAGS) -c learning.cpp

parse.o: parse.cpp parse.h 
	$(CC) $(DEBUG) $(RELEASE) $(CFLAGS) -c parse.cpp

main.o: main.cpp parse.h learning.h
	$(CC) $(DEBUG) $(RELEASE) $(CFLAGS) -c main.cpp

learncluster: main.o parse.o learning.o
	$(CC) $(DEBUG) $(RELEASE) $(CFLAGS) $(LIBS) -o learncluster main.o parse.o learning.o

debug: DEBUG = -g

debug: learncluster

release: RELEASE = -O3

release: learncluster

clean: 
	rm -rf *.o
real_clean: 
	rm -rf *.o *.csv *.txt 	
