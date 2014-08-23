#! /usr/bin/python

import sys
import os
import numpy as np

def dump_matrix(fn, mat):
	with open(fn, "w") as fp:
		rows = mat.shape[0]
		cols = mat.shape[1]
		fp.write("%d %d\n" % (rows, cols))
		for r in xrange(0, rows):
			for c in xrange(0, cols):
				if c > 0:
					fp.write(" ")
				fp.write("%.7f" % (mat[r,c]))
			fp.write("\n")

def gentest(rows, cols):
	mat1 = np.random.randn(rows, cols)
	mat2 = np.random.randn(rows, cols)
	mato = mat1.dot(mat2)
	dirname = "data/%dx%d" % (rows, cols)
	if not os.path.isdir(dirname):
		os.makedirs(dirname)
	dump_matrix("%s/input0.raw" % (dirname), mat1)
	dump_matrix("%s/input1.raw" % (dirname), mat2)
	dump_matrix("%s/output.raw" % (dirname), mato)

def main(argv):
	rows = int(argv[0])
	cols = int(argv[1])
	gentest(rows, cols)

if __name__ == "__main__":
	main(sys.argv[1:])
