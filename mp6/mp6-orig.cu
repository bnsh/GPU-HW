#include "mp6.h"

#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE
__global__ void convolve(
	int imageWidth, int imageHeight, int imageChannels, const float *imageData,
	int maskRows, int maskColumns, const float *maskData,
	float *outputImageData
) {
/*
 * OK, so we're loading the copy first.
 */
	__shared__ float cpy[BLOCKSZ+2*Mask_radius][BLOCKSZ+2*Mask_radius][3];

	int realx = blockIdx.x * blockDim.x + threadIdx.x;
	int realy = blockIdx.y * blockDim.y + threadIdx.y;

	if (
		(0 <= realx) && (realx < imageWidth) &&
		(0 <= realy) && (realy < imageHeight)
	) {
		for (int dy = -Mask_radius; dy <= Mask_radius; ++dy) {
			int srcy = realy + dy;
			int dsty = threadIdx.y + dy + Mask_radius;
			for (int dx = -Mask_radius; dx <= Mask_radius; ++dx) {
				int srcx = realx + dx;
				int dstx = threadIdx.x + dx + Mask_radius;

				if (
					(0 <= srcx) && (srcx < imageWidth) &&
					(0 <= srcy) && (srcy < imageHeight)
				) {
					cpy[dsty][dstx][0] = imageData[srcy*imageWidth*3+srcx*3+0];
					cpy[dsty][dstx][1] = imageData[srcy*imageWidth*3+srcx*3+1];
					cpy[dsty][dstx][2] = imageData[srcy*imageWidth*3+srcx*3+2];
				}
				else {
					cpy[dsty][dstx][0] = 0.0;
					cpy[dsty][dstx][1] = 0.0;
					cpy[dsty][dstx][2] = 0.0;
				}
			}
		}
		__syncthreads();

	// OK, now we have to do the actual convolution.
		float s[3] = {0.0,0.0,0.0};
		for (int dy = -Mask_radius; dy <= Mask_radius; ++dy) {
			int sy = threadIdx.y + dy + Mask_radius;
			int my = dy + Mask_radius;
			for (int dx = -Mask_radius; dx <= Mask_radius; ++dx) {
				int sx = threadIdx.x + dx + Mask_radius;
				int mx = dx + Mask_radius;
				s[0] += cpy[sy][sx][0] * maskData[my * maskColumns + mx];
				s[1] += cpy[sy][sx][1] * maskData[my * maskColumns + mx];
				s[2] += cpy[sy][sx][2] * maskData[my * maskColumns + mx];
			}
		}
		if (s[0] < 0.0) s[0] = 0.0; if (s[0] > 1.0) s[0] = 1.0;
		if (s[1] < 0.0) s[1] = 0.0; if (s[1] > 1.0) s[1] = 1.0;
		if (s[2] < 0.0) s[2] = 0.0; if (s[2] > 1.0) s[2] = 1.0;
		outputImageData[realy * imageWidth * 3 + realx * 3 + 0] = s[0];
		outputImageData[realy * imageWidth * 3 + realx * 3 + 1] = s[1];
		outputImageData[realy * imageWidth * 3 + realx * 3 + 2] = s[2];
	}
}
