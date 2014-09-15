#include <wb.h>

#define TILE_WIDTH (16)

// Compute C = A * B
__global__ void matrixMultiplyShared(const float * A, const float * B, float * C,
						 int numARows, int numAColumns,
						 int numBRows, int numBColumns,
						 int numCRows, int numCColumns) {
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP
	__shared__ float Atile[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Btile[TILE_WIDTH][TILE_WIDTH];

	int Ar = blockIdx.y * blockDim.y + threadIdx.y;
	int Bc = blockIdx.x * blockDim.x + threadIdx.x;

	float Cvalue = 0.0;
	int numtiles = (1+((numAColumns-1)/TILE_WIDTH));
	for (int tile = 0; tile < numtiles; ++tile) {
		int Ac = tile * TILE_WIDTH;
		int Aidx = Ar * numAColumns + (Ac + threadIdx.x);
		int Br = tile * TILE_WIDTH;
		int Bidx = (Br+threadIdx.y) * numBColumns + Bc;

		Atile[threadIdx.x][threadIdx.y] = A[Aidx];
		Btile[threadIdx.x][threadIdx.y] = B[Bidx];
		__syncthreads();
		for (int i = 0; i < TILE_WIDTH; ++i) {
			// Interesting. An array out of bounds _READ_ causes memory faults.
			if (
				(Ar < numARows) &&
				((Ac + i) < numAColumns) &&
				((Br + i) < numBRows) &&
				(Bc < numBColumns)
			) Cvalue += Atile[i][threadIdx.y] * Btile[threadIdx.x][i];
		}
		__syncthreads();
	}

	int Cr = Ar;
	int Cc = Bc;
	int Cidx = Cr * numCColumns + Cc;
	if ((Cr < numCRows) && (Cc < numCColumns)) C[Cidx] = Cvalue;
}
