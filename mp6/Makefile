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

ASSIGNMENT_NUMBER=6
MODE=image

NVSRCS=\
	mp$(ASSIGNMENT_NUMBER).cu \

OBJS=$(CSRCS:C=o) $(NVSRCS:cu=o)

BINS=\
	mp$(ASSIGNMENT_NUMBER)-orig \
	mp$(ASSIGNMENT_NUMBER)-simpler \
	pnmdiff \

TEST_OUTPUT=$(sort $(wildcard data/*/output.ppm))
TEST_INPUTS=$(TEST_OUTPUT:output.ppm=input0.ppm) $(TEST_OUTPUT:output.ppm=input1.csv)
TEST_ORIG= $(TEST_OUTPUT:output.ppm=generated-orig.ppm)
TEST_SIMPLER= $(TEST_OUTPUT:output.ppm=generated-simpler.ppm)

all: $(OBJS) $(BINS)

clean:
	/bin/rm -f $(BINS) $(OBJS)

test: test-orig test-simpler

test-orig: $(TEST_ORIG)

test-simpler: $(TEST_SIMPLER)

push:
	/usr/bin/rsync -avz -e ssh --progress --delete $(HOME)/classes/hetero/mp$(ASSIGNMENT_NUMBER)/ som.hex21.com:classes/hetero/mp$(ASSIGNMENT_NUMBER)/

mp$(ASSIGNMENT_NUMBER)-orig: mp$(ASSIGNMENT_NUMBER).o mp$(ASSIGNMENT_NUMBER)-orig.o
	$(NVCC) $(NVCCFLAGS) $(^) -o $(@) $(NVLIBS)

mp$(ASSIGNMENT_NUMBER)-simpler: mp$(ASSIGNMENT_NUMBER).o mp$(ASSIGNMENT_NUMBER)-simpler.o
	$(NVCC) $(NVCCFLAGS) $(^) -o $(@) $(NVLIBS)

pnmdiff: pnmdiff.o
	$(CC) $(CFLAGS) $(^) -o $(@) -lgd -lnetpbm

data/%/generated-orig.ppm: $(OBJS) $(BINS) data/%/output.ppm data/%/input0.ppm data/%/input1.csv
	./mp$(ASSIGNMENT_NUMBER)-orig -t $(MODE) -i "`printf %s,%s data/$*/input0.ppm data/$*/input1.csv`" -e data/$*/output.ppm -o $(@) | /usr/bin/python ../wb/unwb

data/%/generated-simpler.ppm: $(OBJS) $(BINS) data/%/output.ppm data/%/input0.ppm data/%/input1.csv
	./mp$(ASSIGNMENT_NUMBER)-simpler -t $(MODE) -i "`printf %s,%s data/$*/input0.ppm data/$*/input1.csv`" -e data/$*/output.ppm -o $(@) | /usr/bin/python ../wb/unwb

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) $(^) -o $(@)

%.o: %.C
	$(CC) -c $(CFLAGS) $(INCLUDES) $(^) -o $(@)
