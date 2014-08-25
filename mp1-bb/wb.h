#ifndef WBIMPORT_H
#define WBIMPORT_H


typedef struct {
	int argc;
	char **argv;
} wbArg_t;

#ifdef __cplusplus
extern "C" {
#endif
float *wbImport(const char *fn, int *inputLength);
const char *wbArg_getInputFile(wbArg_t args, int n);
wbArg_t wbArg_read(int argc, char *argv[]);
#ifdef __cplusplus
}
#endif

#endif
