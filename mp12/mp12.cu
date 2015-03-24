#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
	//@@ Insert code to implement vector addition here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) out[idx] = in1[idx] + in2[idx];
}

int main(int argc, char ** argv) {
	wbArg_t args;
	int inputLength;
	float * hostInput1;
	float * hostInput2;
	float * hostOutput;
	float * deviceInput1;
	float * deviceInput2;
	float * deviceOutput;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *) malloc(inputLength * sizeof(float));
	cudaMalloc(&deviceInput1, inputLength * sizeof(float));
	cudaMalloc(&deviceInput2, inputLength * sizeof(float));
	cudaMalloc(&deviceOutput, inputLength * sizeof(float));

	cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	const int bsz = 512;
	dim3 blocksz(bsz, 1, 1);
	dim3 gridsz(((inputLength-1) / bsz) + 1, 1, 1);
	
	vecAdd<<<gridsz, blocksz>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
	cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
	wbSolution(args, hostOutput, inputLength);

	cudaFree(deviceOutput); deviceOutput = NULL;
	cudaFree(deviceInput2); deviceInput2 = NULL;
	cudaFree(deviceInput1); deviceInput1 = NULL;
	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}

