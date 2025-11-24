# Compilers
CC = gcc

CFLAGS += -O3 -Wall -Wextra -march=native

LDFLAGS  += -lm

VPATH = src

OBJ = bin/spmv_spv8.o
MAIN_OBJ = bin/spmv_spv8_main.o

all: bin bin/spmv_spv8

bin:
	mkdir -p bin

bin/spmv_spv8: $(OBJ) $(MAIN_OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(MAIN_OBJ) -o $@ $(LDFLAGS)

bin/%.o: src/%.c | bin
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY : clean
clean :
	-rm bin/*

