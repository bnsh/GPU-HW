CC=g++
CFLAGS=-g3 -O0 -Wall -Werror -DWB_USE_CUSTOM_MALLOC
CLIBS=

# Binesh - Don't ask. Ugh.
comma= ,
empty=
space= $(empty) $(empty)
NVCC=/usr/local/cuda/bin/nvcc
# NVCCFLAGS=-ccbin g++ -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
ARCH=30
NVCCFLAGS=-ccbin g++ --compiler-options $(subst $(space),$(comma),$(CFLAGS)) -m64 -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
NVLIBS=-L../wb/Linux-x86_64/ -lwb -L/usr/local/cuda-6.0/lib64/ -lcudadevrt -lcuda

INCLUDES=-I../wb/

CSRCS=\

ASSIGNMENT_NUMBER=3

NVSRCS=\
	mp$(ASSIGNMENT_NUMBER).cu \

OBJS=$(CSRCS:C=o) $(NVSRCS:cu=o)

BINS=\
	mp$(ASSIGNMENT_NUMBER) \


all: $(OBJS) $(BINS) test

clean:
	/bin/rm -f $(BINS) $(OBJS)

push:
	/usr/bin/rsync -avz -e ssh --progress --delete $(HOME)/classes/hetero/mp$(ASSIGNMENT_NUMBER)/ som.hex21.com:classes/hetero/mp$(ASSIGNMENT_NUMBER)/

test: $(BINS) data/*/input0.raw data/*/input1.raw data/*/output.raw
	for d in data/*; do ./mp$(ASSIGNMENT_NUMBER) -i "$${d}/input0.raw,$${d}/input1.raw" -t matrix -e "$${d}/output.raw" -o "$${d}/mine.raw"; done
	

mp$(ASSIGNMENT_NUMBER): mp$(ASSIGNMENT_NUMBER).o
	$(NVCC) $(NVCCFLAGS) $(^) -o $(@) $(NVLIBS)

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) $(^) -o $(@)

%.o: %.C
	$(CC) -c $(CFLAGS) $(INCLUDES) $(^) -o $(@)
