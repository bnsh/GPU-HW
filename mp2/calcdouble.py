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
			r += 1
		mat = mm
	return mat

def main(argv):
	start = time.time()
	mat1 = read_matrix(argv[0])
	mat2 = read_matrix(argv[1])
	read_done = time.time()
	print "Starting Calculation"
	mato = np.dot(mat1, mat2)
	fin = time.time()
	print "Took %.7f secs to read" % (read_done-start)
	print "Took %.7f secs to multiply" % (fin-read_done)
	print mat1
	print mat1.shape
	print mat2.shape
	print mato.shape

if __name__ == "__main__":
	main(sys.argv[1:])
