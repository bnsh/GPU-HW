// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include	<wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {													\
		cudaError_t err = stmt;											   \
		if (err != cudaSuccess) {											 \
			wbLog(ERROR, "Failed to run stmt ", #stmt);					   \
			wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));	\
			return -1;														\
		}																	 \
	} while(0)

__global__ void total(float * input, float * output, int len) {
	//@@ Load a segment of the input vector into shared memory
	__shared__ float partial[2*BLOCK_SIZE];
	int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	partial[2*threadIdx.x+0] = input[idx+0];
	partial[2*threadIdx.x+1] = input[idx+1];
	__syncthreads();

	//@@ Traverse the reduction tree
	for (int stride = blockDim.x; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			int cidx = 2*blockIdx.x * blockDim.x + threadIdx.x + stride;
			if (cidx < len) partial[threadIdx.x] += partial[threadIdx.x + stride];
		}
		__syncthreads();
	}

	//@@ Write the computed sum of the block to the output vector at the 
	//@@ correct index
	if (threadIdx.x == 0) output[blockIdx.x] = partial[0];
}

int main(int argc, char ** argv) {
	int ii;
	wbArg_t args;
	float * hostInput; // The input 1D list
	float * hostOutput; // The output list
	float * deviceInput;
	float * deviceOutput;
	int numInputElements; // number of elements in the input list
	int numOutputElements; // number of elements in the output list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

	numOutputElements = numInputElements / (BLOCK_SIZE<<1);
	if (numInputElements % (BLOCK_SIZE<<1)) {
		numOutputElements++;
	}
	hostOutput = (float*) malloc(numOutputElements * sizeof(float));

	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
	wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here
	wbCheck(cudaMalloc(&deviceInput, sizeof(float) * numInputElements));
	wbCheck(cudaMalloc(&deviceOutput, sizeof(float) * numOutputElements));

	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here
	wbCheck(cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceOutput, hostOutput, numOutputElements * sizeof(float), cudaMemcpyHostToDevice));

	wbTime_stop(GPU, "Copying input memory to the GPU.");
	//@@ Initialize the grid and block dimensions here
	dim3 blocksz(BLOCK_SIZE, 1, 1);
	dim3 gridsz(((numInputElements-1) / (2*BLOCK_SIZE))+1);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Launch the GPU Kernel here
	total<<<gridsz, blocksz>>>(deviceInput, deviceOutput, numInputElements);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost));

	wbTime_stop(Copy, "Copying output memory to the CPU");

	/********************************************************************
	 * Reduce output vector on the host
	 * NOTE: One could also perform the reduction of the output vector
	 * recursively and support any size input. For simplicity, we do not
	 * require that for this lab.
	 ********************************************************************/
	for (ii = 1; ii < numOutputElements; ii++) {
		hostOutput[0] += hostOutput[ii];
	}

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here
	wbCheck(cudaFree(deviceInput));
	wbCheck(cudaFree(deviceOutput));

	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, 1);

	free(hostInput);
	free(hostOutput);

	return 0;
}

