#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "gd.h"

extern "C" {
#include "pam.h"
}

int main(int argc, char *argv[]) {
    if (argc == 4) {
        const char *fn1 = argv[1];
        const char *fn2 = argv[2];
        const char *fno = argv[3];
        FILE *fp1 = fopen(fn1, "r");
        FILE *fp2 = fopen(fn2, "r");
        if ((fp1) && (fp2)) {
		struct pam p1; memset(&p1, '\0', sizeof(p1));
		struct pam p2; memset(&p2, '\0', sizeof(p2));
		tuple **pixels1 = pnm_readpam(fp1, &p1, sizeof(p1));
		tuple **pixels2 = pnm_readpam(fp2, &p2, sizeof(p2));
		assert(p1.height == p2.height);
		assert(p1.width == p2.width);
		gdImagePtr g1 = gdImageCreateTrueColor(p1.width, p1.height);
		if (fp1) fclose(fp1); fp1 = NULL;
		if (fp2) fclose(fp2); fp2 = NULL;

		for (int i = 0; i < p1.height; ++i) {
			for (int j = 0; j < p2.width; ++j) {
				int c1 = (
					(pixels1[i][j][0] << 16) |
					(pixels1[i][j][1] << 8) |
					(pixels1[i][j][2] << 0)
				) & 0x00ffffff;
				int c2 = (
					(pixels2[i][j][0] << 16) |
					(pixels2[i][j][1] << 8) |
					(pixels2[i][j][2] << 0)
				) & 0x00ffffff;
				g1->tpixels[i][j] = (c1 ^ c2) | 0x00ff000000;
			}
		}
		FILE *fpo = fopen(fno, "w");
		if (fpo) {
			gdImagePng(g1, fpo);
			fclose(fpo); fpo = NULL;
		}
        }
        if (fp1) fclose(fp1); fp1 = NULL;
        if (fp2) fclose(fp2); fp2 = NULL;
    }
    return(0);
}
