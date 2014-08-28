#! /usr/bin/python

import sys
import time
import numpy as np

def read_matrix(fn):
	mat = None
	with open(fn, "r") as fp:
		lines = fp.readlines()
		dims = lines[0].split()
		rows = int(dims[0])
		cols = int(dims[1])
		mm = np.zeros((rows, cols))
		r = 0
		for line in lines[1:]:
			ss = line.split()
			for c in xrange(0, cols):
				mm[r,c] = float(ss[c])
		mat = mm
	return mat

def main(argv):
	mat1 = read_matrix(argv[0])
	mat2 = read_matrix(argv[1])
	start = time.time()
	mato = mat1.dot(mat2)
	fin = time.time()
	print "Took %.7f secs" % (fin-start)

if __name__ == "__main__":
	main(sys.argv[1:])
