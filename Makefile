SHELL=/bin/bash
CC=gcc
CFLAGS=-lpthread
VFLAGS=--leak-check=full --show-leak-kinds=all --track-origins=yes -s

default: pagerank 

pagerank: pagerank.o
	$(CC) $(CFLAGS) $^ -o $@

pagerank.o: pagerank.c

clean:
	rm -f pagerank *.o