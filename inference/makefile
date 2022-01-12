CC=g++
CFLAGS=-I. -O2 -D__AVXACC__ -DLINUX -march=native -mavx -pthread
LIBS= -lm

default: inference

preprocessing.o: preprocessing.cpp classifier.h preprocessing.h
	$(CC) $(CFLAGS) -c preprocessing.cpp

classifier.o: classifier.cpp classifier.h tunables.h 
	$(CC) $(CFLAGS) -c classifier.cpp

tunables.o: tunables.cpp tunables.h 
	$(CC) $(CFLAGS) -c tunables.cpp

main.o: main.cpp classifier.h tunables.h
	$(CC) $(CFLAGS) -c main.cpp

inference: main.o classifier.o tunables.o preprocessing.o
	$(CC) $(CFLAGS) $(LIBS) -o inference main.o classifier.o tunables.o preprocessing.o 

clean: 
	rm -rf *.o
real_clean: 
	rm -rf *.o *.txt *.csv 
