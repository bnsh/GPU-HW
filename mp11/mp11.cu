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

static void computeHistogramOfGrayImage(int width, int height, const unsigned char *grayImage, int *histogram) {
// TODO: Parallelize
	int ii;
	for (ii = 0; ii < 256; ++ii) histogram[ii] = 0;
	for (ii = 0; ii < (width*height); ++ii) histogram[grayImage[ii]]++;
}

static float prob(int width, int height, int x) {
	return(((float)x) / (float)(width * height));
}

static float computeCDFofHistogram(int width, int height, const int *histogram, float *cdf) {
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
	int histogram[HISTOGRAM_LENGTH];
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
	hostInputImageData = wbImage_getData(outputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	castFromImageToUnsignedChar(imageWidth, imageHeight, imageChannels, hostInputImageData, ucharImage);
	convertFromRGBtoGrayScale(imageWidth, imageHeight, ucharImage, grayImage);
	computeHistogramOfGrayImage(imageWidth, imageHeight, grayImage, histogram);
	cdfmin = computeCDFofHistogram(imageWidth, imageHeight, histogram, cdf);
	applyHistogramEqualization(imageWidth, imageHeight, imageChannels, cdf, cdfmin, ucharImage);
	castBackToFloat(imageWidth, imageHeight, imageChannels, ucharImage, hostOutputImageData);

	//@@ insert code here

	wbSolution(args, outputImage);

	//@@ insert code here
	if (grayImage != NULL) free(grayImage); grayImage = NULL;
	if (ucharImage != NULL) free(ucharImage); ucharImage = NULL;

	return 0;
}

