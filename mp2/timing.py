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
	rv = { }
	for i in xrange(2, 1024):
		rv[i] = timing(i)
	print json.dumps(rv, indent=8)

if __name__ == "__main__":
	main(sys.argv[1:])
