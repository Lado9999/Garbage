import numpy as np

class OutOfBoundsError(Exception):
	pass

class Particles:
	'''
	Length in this sim is measured in units of ball diameter
	'''
	def __init__(self, N, L=10, dt=.1):

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

		# speed=1 means that a particle with unit speed
		# traverses the it's length in one step
		self.set_speed()

	def set_speed(self, speed = .5):
		'''
		Set the most probable speed of the particles
		measured in units of 1/dt
		'''
		self._speed = speed / self.dt

	def get_speed(self):
		'''
		Set the most probable speed of the particles
		measured in units of 1/dt
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

	def temperature(self):
		'''
		Returns temperature of the system
		'''
		_T = 0

		if (self.V != 0).any():
			_E = self.energies()
			_expE = np.exp(-_E)
			_T = np.mean(_E * _expE) / np.mean(_expE)

		return _T

	def speeds(self):
		'''
		Return speed distribution of the particles
		'''
		return np.sqrt(self.energies())

	def normalized_speeds(self):
		'''
		Return speed distribution of the particles
		measured in most probable speed
		'''
		return self.speeds()/self._speed

	def randomize(self, opts):
		'''
		Randomize positions and/or velocities of the particles
		opts[string] - options
		'V' - randomize velocities
		'R' - randomize positions
		'RV' or 'VR' - both
		'''
		if 'R' in opts:
			self.R = np.random.uniform(0.5, self.L - 0.5, (2,self.N))

		if 'V' in opts:
			_theta = np.random.uniform(0,2*np.pi,self.N)
			self.V = self._speed * np.vstack((np.cos(_theta), np.sin(_theta)))


	def step(self):
		'''
		Single simulation step
		'''

		self._wall_collision()
		self._ball_collision()

		# modify particle positions according to their new velocities
		self.R += self.V * self.dt


	def _is_arangeble(self):
		'''
		Check if number of particles is a perfect square
		There is currently no way to simulate 50**2 = 2500 particles
		so we just check up to 50
		'''
		range = np.arange(50)
		return range[range**2 == self.N]

	def _wall_collision(self):
		'''
		Check collision with walls and update velocities
		To avoid particles getting stuck in walls
		only update speeds when particles are moving towards the walls
		'''

		mask = self.R[0] < 0.5
		self.V[0][mask] *= np.sign(self.V[0][mask])

		mask = self.R[0] > self.L-0.5
		self.V[0][mask] *= -np.sign(self.V[0][mask])

		mask = self.R[1] < 0.5
		self.V[1][mask] *= np.sign(self.V[1][mask])

		mask = self.R[1] > self.L-0.5
		self.V[1][mask] *= -np.sign(self.V[1][mask])

	def _ball_collision(self):
		'''
		Check collision with eachother and update velocities
		To avoid particles getting stuck together
		only update speeds when particles are moving towards eachother
		'''

		# Xorder = np.argsort(self.R[0], kind='stable')
		#
		# for i in range(self.N):
		# 	for j in range(1,self.N-i):
		# 		pj = Xorder[j]
		# 		pi = Xorder[i]
		# 		if self.R[0, pj] - self.R[0, pi] > 1:
		# 			break
		#
		# 		# Check Collision
		# 		v1, v2 = self.V.T[[pi, pj]]
		# 		r1, r2 = self.R.T[[pi, pj]]
		#
		# 		r = r1 - r2
		# 		v = v1 - v2
		#
		# 		# if balls are moving away from eachother don't collide
		# 		# this avoids stickage
		# 		if np.sign(np.dot(r,v)) < 0:
		# 			rsq = np.dot(r,r)
		# 			if rsq < 1:
		# 				u = r*np.dot(v, r)/rsq
		#
		# 				v1 = v1 - u
		# 				v2 = v2 + u
		#
		# 				self.V.T[[pi, pj]] = np.vstack((v1,v2))


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
			if np.sign(np.dot(r,v)) < 0:
				rsq = np.dot(r,r)
				u = r*np.dot(v, r)/rsq

				v1 = v1 - u
				v2 = v2 + u

				self.V.T[pair] = np.vstack((v1,v2))

if __name__=='__main__':
	pass
