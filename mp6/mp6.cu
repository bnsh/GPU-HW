#include	<wb.h>
#include "mp6.h"


#define wbCheck(stmt) do { \
	cudaError_t err = stmt; \
	if (err != cudaSuccess) { \
		wbLog(ERROR, "Failed to run stmt ", #stmt); \
		wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
		return -1; \
	} \
} while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE
__global__ void convolve(
	int imageWidth, int imageHeight, int imageChannels, const float *imageData,
	int maskRows, int maskColumns, const float *maskData,
	float *outputImageData
);

static void dump(const char *fn, const char *label, int height, int width, int channels, const float *data) {
	FILE *fp = fopen(fn, "w");
	if (fp) {
		fprintf(fp, "%s = {", label);
		for (int i = 0; i < height; ++i) {
			if (i) fprintf(fp, ",");
			fprintf(fp, "\n{");
			for (int j = 0; j < width; ++j) {
				if (j) fprintf(fp, ",");
				fprintf(fp, " %.2f, %.2f, %.2f    ",
					data[i*3*width+j*3+0],
					data[i*3*width+j*3+1],
					data[i*3*width+j*3+2]
				);
			}
			fprintf(fp, "}");
		}
		fprintf(fp, "\n};\n");
		fclose(fp); fp = NULL;
	}
}

int main(int argc, char* argv[]) {
	wbArg_t args;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char * inputImageFile;
	char * inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float * hostInputImageData;
	float * hostOutputImageData;
	float * hostMaskData;
	float * deviceInputImageData;
	float * deviceOutputImageData;
	float * deviceMaskData;

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);
	inputMaskFile = wbArg_getInputFile(args, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");


	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData,
			   hostInputImageData,
			   imageWidth * imageHeight * imageChannels * sizeof(float),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData,
			   hostMaskData,
			   maskRows * maskColumns * sizeof(float),
			   cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");


	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	dim3 blocksz(BLOCKSZ,BLOCKSZ,1);
	dim3 gridsz((((imageWidth-1)/blocksz.x)+1), (((imageHeight-1)/blocksz.y)+1),1);

	convolve<<<gridsz, blocksz>>>(
		imageWidth, imageHeight, imageChannels, deviceInputImageData,
		maskRows, maskColumns, deviceMaskData,
		deviceOutputImageData);

	wbTime_stop(Compute, "Doing the computation on the GPU");


	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData,
			   deviceOutputImageData,
			   imageWidth * imageHeight * imageChannels * sizeof(float),
			   cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	fprintf(stderr, "image = (w=%d x h=%d, c=%d)\n", imageWidth, imageHeight, imageChannels);
	dump("debug/input.m", "input", imageHeight, imageWidth, imageChannels, hostInputImageData);
	dump("debug/output.m", "output", imageHeight, imageWidth, imageChannels, hostOutputImageData);

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
