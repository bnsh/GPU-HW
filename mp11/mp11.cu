// Histogram Equalization

#include	<wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

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

	//@@ Insert more code here

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	wbTime_start(Generic, "Importing data and creating memory on host");
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	hostInputImageData = (float *)(malloc(sizeof(float)*imageWidth*imageHeight*imageChannels));
	hostOutputImageData = (float *)(malloc(sizeof(float)*imageWidth*imageHeight*imageChannels));
	ucharImage = (unsigned char *)(malloc(sizeof(unsigned char)*imageWidth*imageHeight*imageChannels));
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	castFromImageToUnsignedChar(

	//@@ insert code here

	wbSolution(args, outputImage);

	//@@ insert code here
	if (ucharImage != NULL) free(ucharImage); ucharImage = NULL;
	if (hostOutputImageData != NULL) free(hostOutputImageData); hostOutputImageData = NULL;
	if (hostInputImageData != NULL) free(hostInputImageData); hostInputImageData = NULL;

	return 0;
}

