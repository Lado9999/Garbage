#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sys


def ND_sq_lattice(N):
    """
        N_dim_sq_lattice generates all possible moves 
        on an N-dimensional square lattice
        
        parameters:
            N (int) - dimension
        returns:
            (2N,N) numpy.array
    """
    
    return np.vstack((np.eye(N, dtype=int), 
					 -np.eye(N, dtype=int)))


def random_path(N, moves=None):
	"""
	Generates a random path according to allowed moves
	parameters:
		N (int) - path length
		moves (numpy.array) - possible moves
	returns:
		(N,moves.shape[0]) numpy.array
	"""

	return moves[np.random.randint(0,moves.shape[0],N)].cumsum(axis=0)

def plot_path(path):
    if path.shape[0] != 2:
        path = path.T
    plt.scatter(path[0], path[1], alpha=float(sys.argv[2]), s=int(sys.argv[3]))
    plt.axes().set_aspect(1)
    plt.show()

if __name__=='__main__':
		plot_path(random_path(int(sys.argv[1]), ND_sq_lattice(2)))

