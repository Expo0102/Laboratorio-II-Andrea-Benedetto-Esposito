SHELL=/bin/bash
CC=gcc
CFLAGS=-lpthread

default: pagerank 

pagerank: pagerank.o
	$(CC) $(CFLAGS) $^ -o $@

pagerank.o: pagerank.c

clean:
	rm -f pagerank *.o
