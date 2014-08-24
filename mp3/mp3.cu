
#include    <wb.h>

#define TILEWIDTH (16)
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP
	__shared__ float mA[TILEWIDTH][TILEWIDTH];
	__shared__ float mB[TILEWIDTH][TILEWIDTH];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
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
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

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

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 blocksz(TILEWIDTH,TILEWIDTH,1);
    dim3 gridsz(((numCRows-1)/blocksz.x)+1,((numCColumns-1)/blocksz.y)+1,1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<blocksz, gridsz>>>(deviceA, deviceB, deviceC,
        numARows, numAColumns,
        numBRows, numBColumns,
        numCRows, numCColumns);

    cudaThreadSynchronize();
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

    free(hostA); hostA = NULL;
    free(hostB); hostB = NULL;
    free(hostC); hostC = NULL;

    return 0;
}

