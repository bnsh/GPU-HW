#include <stdio.h>
#include <stdlib.h>
#include "wb.h"

int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength, i;
	float * hostInput1;
	float * hostInput2;

	args = wbArg_read(argc, argv);

	//wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);


	for (i = 0; i < inputLength; ++i) {
		printf("%.7f	+ %.7f	= %.7f\n", hostInput1[i], hostInput2[i], hostInput1[i] + hostInput2[i]);
	}
	free(hostInput2); hostInput2 = NULL;
	free(hostInput1); hostInput1 = NULL;
	return(0);
}
