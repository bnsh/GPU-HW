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
	__shared__ float cpy[BLOCKSZ][BLOCKSZ][3];

	int realx = blockIdx.x * blockDim.x + threadIdx.x;
	int realy = blockIdx.y * blockDim.y + threadIdx.y;

	if (
		(0 <= realx) && (realx < imageWidth) &&
		(0 <= realy) && (realy < imageHeight)
	) {
		cpy[threadIdx.y][threadIdx.x][0] = imageData[realy*imageWidth*3 + realx*3 + 0];
		cpy[threadIdx.y][threadIdx.x][1] = imageData[realy*imageWidth*3 + realx*3 + 1];
		cpy[threadIdx.y][threadIdx.x][2] = imageData[realy*imageWidth*3 + realx*3 + 2];
		__syncthreads();

	// OK, now we have to do the actual convolution.
		float s[3] = {0.0,0.0,0.0};
		for (int dy = -Mask_radius; dy <= Mask_radius; ++dy) {
			int ry = realy + dy;
			int sy = threadIdx.y + dy;
			int my = dy + Mask_radius;
			for (int dx = -Mask_radius; dx <= Mask_radius; ++dx) {
				int rx = realx + dx;
				int sx = threadIdx.x + dx;
				int mx = dx + Mask_radius;

				if (
					((0 <= sy) && (sy < BLOCKSZ)) &&
					((0 <= sx) && (sx < BLOCKSZ)) &&
					((0 <= ry) && (ry < imageHeight)) &&
					((0 <= rx) && (rx < imageWidth))
				) {
					s[0] += maskData[my * maskColumns + mx] * cpy[sy][sx][0];
					s[1] += maskData[my * maskColumns + mx] * cpy[sy][sx][1];
					s[2] += maskData[my * maskColumns + mx] * cpy[sy][sx][2];
				}
				else if (
					((0 <= ry) && (ry < imageHeight)) &&
					((0 <= rx) && (rx < imageWidth))
				) {
					s[0] += maskData[my * maskColumns + mx] * imageData[ry*imageWidth*3 + rx*3 + 0];
					s[1] += maskData[my * maskColumns + mx] * imageData[ry*imageWidth*3 + rx*3 + 1];
					s[2] += maskData[my * maskColumns + mx] * imageData[ry*imageWidth*3 + rx*3 + 2];
				}
				else {
					s[0] += maskData[my * maskColumns + mx] * 0;
					s[1] += maskData[my * maskColumns + mx] * 0;
					s[2] += maskData[my * maskColumns + mx] * 0;
				}
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
