import numpy as np

class OutOfBoundsError(Exception):
	pass

class Particles:
	'''
	Every length in this simulation is normalized to the ball diameter
	'''
	def __init__(self, N, L=10, dt=.1, speed = .005):

		# number of balls
		self.N = N

		# size of system
		self.L = L

		# time_step of the simulation
		self.dt = dt

		# positions of the ball centers
		self.R = None

		# velocities of the balls
		self.V = None

		# We measure speed in terms of L
		# speed=1 means that a particle with unit speed
		# traverses the whole system in one step
		self.set_speed(speed)

	def set_speed(self, speed):
		'''
		Set speed of the particles measured in units of L/dt
		'''
		self._speed = (self.L / self.dt) * speed

	def get_speed(self):
		'''
		Getter and setter for speed is used to avoid direc interaction
		with self._speed variable
		'''
		return self._speed

	def freeze(self):
		'''
		Set all velocities to zero
		'''
		self.V = np.zeros((2, self.N))

	def arange_in_grid(self, spacing=0):
		'''
		Arange particles in a centered square grid with given grid spacing.
		Will only arange if particle number is a perfect square
		and the grid fits inside the axis bounds.
		'''
		_num = self._is_arangeble()

		if _num.shape != 0:
			_num = _num[0]
			if (spacing + 1) * _num > self.L + 1:
				raise OutOfBoundsError('Ball grid is out of bounds')

			_X = np.arange(_num)*(spacing + 1) + (self.L + spacing + 1 - _num*(spacing + 1))/2

			self.R = np.vstack(np.meshgrid(_X, _X)).reshape(2, self.N)

		else:
			raise ValueError('Can\'t be arranged, Not a perfect square')

	def energies(self):
		'''
		Return energy distribution of the particles
		'''
		return (self.V*self.V).sum(axis=0)

	# TODO Recheck the theory --> kT = \sum E_n*exp(-E_n) / \sum E_n
	def temperature(self):
		'''
		Returns effective temperature of the system
		'''
		_T = 0

		if (self.V != 0).any():
			_E = self.energies()
			_T = np.mean(_E*np.exp(-_E))/np.mean(_E)	

		return _T

	def speeds(self):
		'''
		Return speed distribution of the particles
		'''
		return np.sqrt(self.energies())

	def normalized_speeds(self):
		'''
		Return normalized speed distribution of the particles
		'''
		_factor = 1.0 # TODO change the fator
		return self.speeds()/self._speed

	def randomize(self, opts):
		'''
		Randomize positions and/or velocities of the particles
		opts is a string of options
		'V' - randomizes velocities
		'R' - randomizes positions
		'RV' or 'VR' - both
		'''
		if 'R' in opts:
			self.R = np.random.uniform(0.5, self.L - 0.5, (2,self.N))

		if 'V' in opts:
			_theta = np.random.uniform(0,2*np.pi,self.N)
			self.V = self._speed * np.vstack((np.cos(_theta), np.sin(_theta)))		

	# TODO WRITE  
	def equilibrate(self):
		'''
		Runs the simulation untill the temperature fluctuations dissipate
		'''
		pass
			

	def step(self):

		# hacky collision with walls
		# Only update velocities if they are antiparallel to wall normals
		# so that particles don't get stuck to walls
		self.V[0][self.R[0] < 0.5] *= np.sign(self.V[0][self.R[0] < 0.5])
		self.V[0][self.R[0] > self.L-0.5] *= -np.sign(self.V[0][self.R[0] > self.L-0.5])
		self.V[1][self.R[1] < 0.5] *= np.sign(self.V[1][self.R[1] < 0.5])
		self.V[1][self.R[1] > self.L-0.5] *= -np.sign(self.V[1][self.R[1] > self.L-0.5])

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
			v = v1 - v2

			# if balls are moving away from eachother don't collide
			# this avoids stickage
			if np.sign(np.dot(r,v)) < 0:
				rsq = np.dot(r,r)
				u = r*np.dot(v, r)/rsq

				v1 = v1 - u
				v2 = v2 + u

				self.V.T[pair] = np.vstack((v1,v2))

		# modify particle positions according to their velocities
		self.R += self.V*self.dt

	def _is_arangeble(self):
		range = np.arange(50)
		return range[range**2 == self.N]

if __name__=='__main__':
	pass
