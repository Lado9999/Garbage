import numpy as np
from matplotlib import pyplot as plt

class Particles:
    '''
        Every length in this simulation is normalized to the ball diameter
    '''
    def __init__(self, N, system_size=10, dt=.1, speed = .005):
        
        # number of balls
        self.N = N

        # system parameters
        if type(system_size) is tuple:
                self.W, self.H = system_size
        else:
                self.W, self.H = system_size, system_size

        self.dt = dt # time_step of the simulation

        # positions of the ball centers
        self.R = None
        # self.randomize('R')
        self.arange_in_grid()

        # velocities of the balls
        self.V = np.zeros((2, N))

        # We measure speed in terms of system_size
        # speed=1 means that a particle with unit speed
        # traverses the whole system in unit time
        self.speed = min(self.H, self.W) * speed / dt

    def arange_in_grid(self):
        X = np.linspace(0,1,int(np.sqrt(self.N)))*self.W*.8 + 10
        Y = np.linspace(0,1,int(np.sqrt(self.N)))*self.H*.8 + 10
        self.R = np.vstack(np.meshgrid(X,Y)).reshape(2, self.N)


    def energies(self):
        return (self.V*self.V).sum(axis=0)

    def speeds(self):
        return np.sqrt(self.energies())

    def randomize(self, opts):
    '''
        Randomize positions and/or velocities of the balls
        TODO randomize positions in a way that the balls don't overlap
    '''
    if 'R' in opts:
        if self.R is None:
            self.R = np.zeros((2, self.N))

            self.R[0] = np.random.uniform(0.5, self.W - 0.5, self.N)
            self.R[1] = np.random.uniform(0.5, self.H - 0.5, self.N)

    if 'V' in opts:
        self.V = self.speed * (2*np.random.rand(*self.V.shape) - 1)

    def step(self):

        # collision with walls
        self.V[0][(self.R[0] < 0.5) | (self.R[0] > self.W-0.5)] *= -1
        self.V[1][(self.R[1] < 0.5) | (self.R[1] > self.H-0.5)] *= -1

        # check collision with eachother
        # TODO implement a more efficient algorithm
        dst = self.R[:,np.newaxis,:] - self.R[:,:,np.newaxis]
        dst = (dst*dst).sum(axis=0) < 1 # diameter**2
        for i,d in enumerate(dst):
            d[:i+1] = False
        pairs = np.argwhere(dst)
        for pair in pairs:
            v1, v2 = self.V.T[pair]
            r1, r2 = self.R.T[pair]

            r = r1 - r2
            rsq = np.dot(r,r)
            u = r*np.dot(v1 - v2, r)/rsq

            v1 = v1 - u
            v2 = v2 + u

            self.V.T[pair] = np.vstack((v1,v2))


        # modify particle positions according to their velocities
        self.R += self.V*self.dt

if __name__=='__main__':
    pass
