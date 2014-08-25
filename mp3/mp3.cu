
#include    <wb.h>
#include "cuPrintf.cu"

#define TILE_WIDTH (16)
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(const float * A, const float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP
	__shared__ float Atile[2 * TILE_WIDTH * TILE_WIDTH];
	float *Btile = Atile + TILE_WIDTH * TILE_WIDTH;

	cuPrintf("Before A = %dx%d B=%dx%d C=%dx%d C=%p\n",
		numARows, numAColumns,
		numBRows, numBColumns,
		numCRows, numCColumns, C);
	int Ar = blockIdx.y * blockDim.y;
	int Bc = blockIdx.x * blockDim.x;

	float Cvalue = 0.0;
	for (int tile = 0; tile < (1+((numAColumns-1)/TILE_WIDTH)); ++tile) {
		int Ac = tile * TILE_WIDTH;
		int Aidx = (Ar+threadIdx.y) * numAColumns + (Ac + threadIdx.x);
		int Br = tile * TILE_WIDTH;
		int Bidx = (Br+threadIdx.y) * numBColumns + (Bc + threadIdx.x);

		Atile[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[Aidx];
		Btile[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[Bidx];
		__syncthreads();
		for (int i = 0; i < TILE_WIDTH; ++i) Cvalue += Atile[threadIdx.y * TILE_WIDTH + threadIdx.x+i] * Btile[(threadIdx.y+i) * TILE_WIDTH + threadIdx.x];
		__syncthreads();
	}

	int Cr = Ar;
	int Cc = Bc;
	int Cidx = (Cr + threadIdx.y) * numCColumns + (Cc + threadIdx.x);
	cuPrintf("After A = %dx%d B=%dx%d C=%dx%d C=%p\n",
		numARows, numAColumns,
		numBRows, numBColumns,
		numCRows, numCColumns, C);
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA = NULL; // The A matrix
    float * hostB = NULL; // The B matrix
    float * hostC = NULL; // The output C matrix
    float * deviceA = NULL;
    float * deviceB = NULL;
    float * deviceC = NULL;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *)(malloc(sizeof(float) * numCRows * numCColumns));
    for (int i = 0; i < (numCRows * numCColumns); ++i) hostC[i] = i;
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc(&deviceA, (sizeof(float) * numARows * numAColumns)));
    wbCheck(cudaMalloc(&deviceB, (sizeof(float) * numBRows * numBColumns)));
    wbCheck(cudaMalloc(&deviceC, (sizeof(float) * numCRows * numCColumns)));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA, hostA, (sizeof(float) * numARows * numAColumns), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, (sizeof(float) * numBRows * numBColumns), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceC, hostC, (sizeof(float) * numBRows * numBColumns), cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 blocksz(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridsz(((numCColumns-1)/blocksz.x)+1,((numCRows-1)/blocksz.y)+1,1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    cudaPrintfInit();
    matrixMultiplyShared<<<gridsz, blocksz>>>(deviceA, deviceB, deviceC,
        numARows, numAColumns,
        numBRows, numBColumns,
        numCRows, numCColumns);

    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostC, deviceC, (sizeof(float) * numCRows * numCColumns), cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceC)); deviceC = NULL;
    wbCheck(cudaFree(deviceB)); deviceB = NULL;
    wbCheck(cudaFree(deviceA)); deviceA = NULL;

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostC); hostC = NULL;
    free(hostB); hostB = NULL;
    free(hostA); hostA = NULL;

    return 0;
}

