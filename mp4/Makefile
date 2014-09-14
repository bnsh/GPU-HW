CC=g++
CFLAGS=-g3 -O3 -Wall -Werror -DWB_USE_CUSTOM_MALLOC
CLIBS=

# Binesh - Don't ask. Ugh.
comma= ,
empty=
space= $(empty) $(empty)
NVCC=/usr/local/cuda/bin/nvcc
# NVCCFLAGS=-ccbin g++ -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
ARCH=30
NVCCFLAGS=-ccbin g++ --compiler-options $(subst $(space),$(comma),$(CFLAGS)) -m64 -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
NVLIBS=-L../wb/Linux-x86_64/ -lwb -lcuda

INCLUDES=-I../wb/

CSRCS=\

ASSIGNMENT_NUMBER=4
MODE=vector

NVSRCS=\
	mp$(ASSIGNMENT_NUMBER).cu \

OBJS=$(CSRCS:C=o) $(NVSRCS:cu=o)

BINS=\
	mp$(ASSIGNMENT_NUMBER) \

TEST_OUTPUT=$(sort $(wildcard data/*/output.raw))
TEST_INPUTS=$(TEST_OUTPUT:output.raw=input0.raw)
TEST_ACTUAL=$(TEST_OUTPUT:output.raw=generated.raw)

all: $(OBJS) $(BINS)

clean:
	/bin/rm -f $(BINS) $(OBJS)

test: $(TEST_ACTUAL)

push:
	/usr/bin/rsync -avz -e ssh --progress --delete $(HOME)/classes/hetero/mp$(ASSIGNMENT_NUMBER)/ som.hex21.com:classes/hetero/mp$(ASSIGNMENT_NUMBER)/

mp$(ASSIGNMENT_NUMBER): mp$(ASSIGNMENT_NUMBER).o
	$(NVCC) $(NVCCFLAGS) $(^) -o $(@) $(NVLIBS)

data/%/generated.raw: $(OBJS) $(BINS) data/%/output.raw data/%/input0.raw
	./mp$(ASSIGNMENT_NUMBER) -t $(MODE) -i data/$*/input0.raw -e data/$*/output.raw -o $(@) | /usr/bin/python ../wb/unwb

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) $(^) -o $(@)

%.o: %.C
	$(CC) -c $(CFLAGS) $(INCLUDES) $(^) -o $(@)
