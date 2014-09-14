NVCC=/usr/local/cuda-6.0/bin/nvcc
NVCCFLAGS=-g -O3 -ccbin g++ -m64    -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
CC=gcc
CFLAGS=-g3 -O3 -Wall -Werror

CSRCS=\
	wb.c \
	test.c \

NVSRCS=\
	mp1.cu \
	cuPrintftest.cu \

OBJS=$(CSRCS:c=o) $(NVSRCS:cu=o)

BINS=\
	test \
	cuPrintftest \
	mp1 \

all: $(OBJS) $(BINS)

clean:
	/bin/rm -f $(BINS) $(OBJS)

cuPrintftest: cuPrintftest.o
	$(NVCC) $(NVCCFLAGS) $(^) -o $(@) $(LIBS)

test: test.o wb.o
	$(CC) $(CFLAGS) $(^) -o $(@) $(LIBS)

mp1: mp1.o wb.o
	$(NVCC) $(NVCCFLAGS) $(^) -o $(@) $(LIBS)

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(^) -o $(@)

%.o: %.c
	$(CC) -c $(CFLAGS)  $(^) -o $(@)
