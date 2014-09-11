#! /usr/bin/python

import sys
import time
import numpy as np
import json

def timing(i):
	mat1 = np.float32(np.random.rand(i,i))
	mat2 = np.float32(np.random.rand(i,i))
	start = time.time()
	read_done = time.time()
	mato = np.dot(mat1, mat2)
	fin = time.time()
	return (fin - start)
	

def main(argv):
	iters = 4096
	if len(argv) > 0:
		iters = atoi(argv[0])
	rv = { }
	for i in xrange(1, 1+iters):
		rv[i] = timing(i)
		sys.stderr.write("%5d/%5d %13.7f\r" % (i, iters, rv[i]))
	print json.dumps(rv, indent=8)

if __name__ == "__main__":
	main(sys.argv[1:])
