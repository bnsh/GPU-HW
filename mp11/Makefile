CC=g++
CFLAGS=-g3 -O3 -Wall -Werror -DWB_USE_CUSTOM_MALLOC
CLIBS=

# Binesh - Don't ask. Ugh.
comma= ,
empty=
space= $(empty) $(empty)
NVCC=/usr/local/cuda/bin/nvcc
# NVCCFLAGS=-ccbin g++ -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
ARCHS=20 30 32 35 50 52
NVCCFLAGS=-ccbin g++ --compiler-options $(subst $(space),$(comma),$(CFLAGS)) -m64 $(foreach ARCH,$(ARCHS),-gencode arch=compute_$(ARCH),code=sm_$(ARCH))
NVLIBS=-L../wb/Linux-x86_64/ -lwb -lcuda

INCLUDES=-I../wb/

CSRCS=\

ASSIGNMENT_NUMBER=11
MODE=image

NVSRCS=\
	mp$(ASSIGNMENT_NUMBER).cu \

OBJS=$(CSRCS:C=o) $(NVSRCS:cu=o)

BINS=\
	mp$(ASSIGNMENT_NUMBER) \
	mp$(ASSIGNMENT_NUMBER)-serial \

TEST_OUTPUT=$(sort $(wildcard data/*/output.ppm))
TEST_INPUTS=$(TEST_OUTPUT:output.ppm=input.ppm)
TEST_ACTUAL=$(TEST_OUTPUT:output.ppm=generated.ppm)

all: $(OBJS) $(BINS)

clean:
	/bin/rm -f $(BINS) $(OBJS) mp11-serial.o

test: $(TEST_ACTUAL)

test-serial: $(TEST_ACTUAL:generated.ppm=serial.ppm)

push:
	/usr/bin/rsync -avz -e ssh --progress --delete $(HOME)/classes/hetero/mp$(ASSIGNMENT_NUMBER)/ som.hex21.com:classes/hetero/mp$(ASSIGNMENT_NUMBER)/

mp$(ASSIGNMENT_NUMBER): mp$(ASSIGNMENT_NUMBER).o
	$(NVCC) $(NVCCFLAGS) $(^) -o $(@) $(NVLIBS)

mp$(ASSIGNMENT_NUMBER)-serial: mp$(ASSIGNMENT_NUMBER)-serial.o
	$(NVCC) $(NVCCFLAGS) $(^) -o $(@) $(NVLIBS)

data/%/generated.ppm: $(OBJS) $(BINS) data/%/output.ppm data/%/input.ppm
	./mp$(ASSIGNMENT_NUMBER) -t $(MODE) -i data/$*/input.ppm -e data/$*/output.ppm -o $(@) | /usr/bin/python ../wb/unwb

data/%/serial.ppm: $(OBJS) $(BINS) data/%/output.ppm data/%/input.ppm
	./mp$(ASSIGNMENT_NUMBER)-serial -t $(MODE) -i data/$*/input.ppm -e data/$*/output.ppm -o $(@) | /usr/bin/python ../wb/unwb

%-serial.o: %.cu
	$(NVCC) -DSERIAL -c $(NVCCFLAGS) $(INCLUDES) $(^) -o $(@)

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) $(^) -o $(@)

%.o: %.C
	$(CC) -c $(CFLAGS) $(INCLUDES) $(^) -o $(@)
