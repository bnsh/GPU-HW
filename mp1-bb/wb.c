#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "wb.h"

float *wbImport(const char *fn, int *inputLength) {
	float *retVal = NULL;
	FILE *fp = fopen(fn, "r");
	if (fp) {
		int i;
		assert(1 == fscanf(fp, "%d", inputLength));
		float *data = (float *)malloc((sizeof(float) * (*inputLength)));
		printf("%s: %d\n", fn, (*inputLength));
		for (i = 0; i < (*inputLength); ++i) {
			assert(1 == fscanf(fp, "%f", &data[i]));
		}
		fclose(fp);
		retVal = data;
	}
	return(retVal);
}

const char *wbArg_getInputFile(wbArg_t args, int n) { assert((1+n) < args.argc); printf("%d: %s\n", n, args.argv[1+n]); return(args.argv[1+n]); }

wbArg_t wbArg_read(int argc, char *argv[]) {
	wbArg_t rv;
	rv.argc = argc;
	rv.argv = argv;
	return rv;
}
