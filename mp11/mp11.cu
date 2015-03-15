// Histogram Equalization

#include	<wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

static void castFromImageToUnsignedChar(int width, int height, int channels, const float *inputImage, unsigned char *ucharImage) {
// TODO: Parallelize
	int ii = 0;
	for (ii = 0; ii < (width*height*channels); ++ii) {
		ucharImage[ii] = (unsigned char)(255*inputImage[ii]);
	}
}

static void convertFromRGBtoGrayScale(int width, int height, const unsigned char *ucharImage, unsigned char *grayImage) {
// TODO: Parallelize
	int ii, jj;
	for (ii = 0; ii < height; ++ii) {
		for (jj = 0; jj < width; ++jj) {
			int idx = ii * width + jj;
			int r = ucharImage[3 * idx + 0];
			int g = ucharImage[3 * idx + 1];
			int b = ucharImage[3 * idx + 2];
			grayImage[idx] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
		}
	}
}

#ifdef SERIAL
static void computeHistogramOfGrayImage(int width, int height, const unsigned char *grayImage, unsigned int *histogram) {
// TODO: Parallelize
	int ii;
	for (ii = 0; ii < 256; ++ii) histogram[ii] = 0;
	for (ii = 0; ii < (width*height); ++ii) histogram[grayImage[ii]]++;
}
#else
__global__ void computeHistogramOfGrayImage(int width, int height, const unsigned char *grayImage, unsigned int *histogram) {
	__shared__ unsigned int private_histo[256];
	if (threadIdx.x < 256) private_histo[threadIdx.x] = 0;
	__syncthreads();
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int sz = width * height;
	while (i < sz) {
		unsigned int ss = (unsigned int)grayImage[i];
		atomicAdd(&private_histo[ss], 1);
		i += stride;
	}
	__syncthreads();
	if (threadIdx.x < 256) atomicAdd(&histogram[threadIdx.x], private_histo[threadIdx.x]);
}
#endif

static float prob(int width, int height, int x) {
	return(((float)x) / (float)(width * height));
}

static float computeCDFofHistogram(int width, int height, const unsigned int *histogram, float *cdf) {
	int ii;
	float cdfmin;
	cdf[0] = prob(width, height, histogram[0]);
	cdfmin = cdf[0];
	for (ii = 1; ii < 256; ++ii) {
		cdf[ii] = cdf[ii-1] + prob(width, height, histogram[ii]);
		if (cdfmin > cdf[ii]) cdfmin = cdf[ii];
	}
	return(cdfmin);
}

static unsigned char correct_color(const float *cdf, float cdfmin, int val) {
	int rv = (255*(cdf[val] - cdfmin) / (1.0 - cdfmin));
	if (rv < 0) rv = 0;
	if (rv > 255) rv = 255;
	return((unsigned char)rv);
}

static void applyHistogramEqualization(int width, int height, int channels, const float *cdf, float cdfmin, unsigned char *ucharImage) {
// TODO: Parallelize
	int ii;
	for (ii = 0; ii < (width * height * channels); ++ii) ucharImage[ii] = correct_color(cdf, cdfmin, ucharImage[ii]);
}

static void castBackToFloat(int width, int height, int channels, const unsigned char *ucharImage, float *outputImage) {
// TODO: Parallelize
	int ii;
	for (ii = 0; ii < (width * height * channels); ++ii)
		outputImage[ii] = (float)(ucharImage[ii] / 255.0);
}

static void dump(const char *fn, int imageWidth, int imageHeight, const unsigned char *grayImage, const unsigned int *histogram) {
		FILE *fp = fopen(fn, "w");
		if (fp) {
			fprintf(fp, "grayImage = {");
			for (int i = 0; i < imageWidth*imageHeight; ++i) {
				if (i) fprintf(fp, ",");
				fprintf(fp, "\n%d", grayImage[i]);
			}
			fprintf(fp, "\n}\n");

			fprintf(fp, "histogram = {");
			for (int i = 0; i < 256; ++i) {
				if (i) fprintf(fp, ",");
				fprintf(fp, "\n%d", histogram[i]);
			}
			fprintf(fp, "\n}\n");

			fclose(fp); fp = NULL;
		}
}

int main(int argc, char ** argv) {
	wbArg_t args;
	int imageWidth;
	int imageHeight;
	int imageChannels;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float * hostInputImageData;
	float * hostOutputImageData;
	const char * inputImageFile;
	unsigned char *ucharImage = NULL;
	unsigned char *grayImage = NULL;
	unsigned char *devicegrayImage = NULL;
	unsigned int histogram[HISTOGRAM_LENGTH];
	unsigned int *devicehistogram = NULL;
	float cdf[HISTOGRAM_LENGTH];
	float cdfmin;

	//@@ Insert more code here

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	wbTime_start(Generic, "Importing data and creating memory on host");
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	ucharImage = (unsigned char *)(malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels));
	grayImage = (unsigned char *)(malloc(sizeof(unsigned char)*imageWidth*imageHeight));
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	cudaMalloc((void **)&devicehistogram, 256 * sizeof(int));
	cudaMalloc((void **)&devicegrayImage, imageWidth * imageHeight * sizeof(unsigned char));
	for (int i = 0; i < 256; ++i) histogram[i] = 0;
	cudaMemcpy(devicehistogram, histogram, 256 * sizeof(int), cudaMemcpyHostToDevice);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	castFromImageToUnsignedChar(imageWidth, imageHeight, imageChannels, hostInputImageData, ucharImage);
	convertFromRGBtoGrayScale(imageWidth, imageHeight, ucharImage, grayImage);
	cudaMemcpy(devicegrayImage, grayImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
#ifdef SERIAL
	dump("/tmp/mp11-serial-before.txt", imageWidth, imageHeight, grayImage, histogram);
	computeHistogramOfGrayImage(imageWidth, imageHeight, grayImage, histogram);
	dump("/tmp/mp11-serial-after.txt", imageWidth, imageHeight, grayImage, histogram);
#else
	dump("/tmp/mp11-gpu-before.txt", imageWidth, imageHeight, grayImage, histogram);
	int sz = imageWidth * imageHeight;
	dim3 blocksz(256, 1, 1);
	dim3 gridsz(((sz-1) / blocksz.x)+1, 1, 1);
	fprintf(stderr, "     sz = %d\n", sz);
	fprintf(stderr, "blocksz = { x=%d, y=%d, z=%d }\n", blocksz.x, blocksz.y, blocksz.z);
	fprintf(stderr, " gridsz = { x=%d, y=%d, z=%d }\n", gridsz.x, gridsz.y, gridsz.z);
	computeHistogramOfGrayImage<<<gridsz, blocksz>>>(imageWidth, imageHeight, devicegrayImage, devicehistogram);
	cudaMemcpy(histogram, devicehistogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	dump("/tmp/mp11-gpu-after.txt", imageWidth, imageHeight, grayImage, histogram);
#endif
	cdfmin = computeCDFofHistogram(imageWidth, imageHeight, histogram, cdf);
	applyHistogramEqualization(imageWidth, imageHeight, imageChannels, cdf, cdfmin, ucharImage);
	castBackToFloat(imageWidth, imageHeight, imageChannels, ucharImage, hostOutputImageData);

	//@@ insert code here

	wbSolution(args, outputImage);

	//@@ insert code here
	if (devicegrayImage != NULL) cudaFree(devicegrayImage); devicegrayImage = NULL;
	if (devicehistogram != NULL) cudaFree(devicehistogram); devicehistogram = NULL;
	if (grayImage != NULL) free(grayImage); grayImage = NULL;
	if (ucharImage != NULL) free(ucharImage); ucharImage = NULL;

	return 0;
}

