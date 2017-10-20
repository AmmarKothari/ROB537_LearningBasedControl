import numpy as np
from pprint import pprint
import pdb
import matplotlib.pyplot as plt
import copy
import multiprocessing as mp
import pygame


class SlotMachine(object):
	# defines a single slot machine in N armed bandit problem
	# Gaussian distribution of rewards
	def __init__(self, mean, variance, dist = 'Gaussian'):
		self.mean = float(mean)
		self.variance = float(variance)
		self.dist = dist

	def reward(self, n=1):
		if self.dist.lower() == 'Gaussian'.lower():
			out = np.random.normal(loc=self.mean, scale=self.variance, size=n)
		if self.dist.lower() == 'Uniform'.lower():
			out = np.random.uniform(low=self.mean-self.variance, high=self.mean+self.variance, size=n)
		# out[out<0] = 0 #set all negative values to zero
		return out


class Bandit(object):
	#class to define the bandit
	def __init__(self, means, variances):
		if len(means) != len(variances):
			raise(ValueError('Mean and Variance arrays are not equal'))
		self.num_actions = len(means)
		self.means = means
		self.variances = variances
		self.slots = [SlotMachine(x,y) for x,y in zip(means, variances)]

	def testSlots(self, N=100):
		for i in range(N):
			pprint('Slot Output:%s' %np.array([s.reward() for s in self.slots]).flatten())


class BanditPlayer(object):
	# initializes and tests N-armed bandit
	def __init__(self, Bandit):
		self.Bandit = Bandit
		self.num_actions = Bandit.num_actions
		#initialize table with something
		self.initalizeZero()
		self.V_settings()

	def V_settings(self, alpha=None):
		self.alpha = alpha

	def initalizeZero(self):
		self.ActionValueTable = np.zeros(self.num_actions)

	def set_AV_Table(self, AV):
		self.ActionValueTable = AV

	def initializeRandom(self):
		self.ActionValueTable = np.random.rand(self.num_actions)

	def initalizeOptimistic(self, opt_val = 1e1):
		self.ActionValueTable = np.zeros(self.num_actions) + opt_val


	def EpsilonGreedyChooseAction(self, epsilon):
		if np.random.rand(1) < epsilon:
			# choose at random
			action = np.random.randint(self.num_actions)
		else:
			# always chooses the first max that it finds in list
			action = np.argmax(self.ActionValueTable)
		return action

	def V_update(self, action, r):
		self.ActionValueTable[action] = self.ActionValueTable[action] + self.alpha * (r - self.ActionValueTable[action])

	def AV_update(self, AV):
		# is this the correct update
		self.ActionValueTable = 0.5 * (self.ActionValueTable + AV)

	def converge(self, steps, epsilon, N):
			if self.alpha is None:
				raise(ValueError('Must set alpha value first with V_settings'))
			for i in range(N):
				for s in range(steps):
					a = self.EpsilonGreedyChooseAction(epsilon)
					r = self.Bandit.slots[a].reward()
					self.V_update(a, r)

	def test_VA(self, steps, N):
		epsilon = 0 # or should i keep it the same??
		rs = []
		for i in range(N):
			r = 0
			for s in range(steps):
				a = self.EpsilonGreedyChooseAction(epsilon)
				r =+ self.Bandit.slots[a].reward()
			rs.append(r)
		print('Average Reward: %s, StDev: %s' %(np.mean(rs), np.std(rs)))
		return rs

	def plotVATable(self, VA, show=True):
		width = 1.0/len(VA)
		ax = plt.subplot(1,1,1)
		f = ax.get_figure()
		ind = np.arange(self.num_actions)

		ax.bar(ind, VA, width, color='r', label='ValueAction')
		ax.bar(ind+width, self.Bandit.means, width, color='b', label='BanditMeans')
		# ax.bar(ind+width, self.Bandit.means, color='b', yerr=self.Bandit.variances)
		plt.legend()
		if show: plt.show()
		return ax

class BanditPlayerMP(BanditPlayer):
	def __init__(self, Bandit):
		super(BanditPlayerMP, self).__init__(Bandit)

	def converge_multi(self, steps, epsilon, N):
		# rewrite this to be asynchronous updates!
		# create processes
		# evertime you get something back, update parameters and spawn a new process
		if self.alpha is None:
			raise(ValueError('Must set alpha value first with V_settings'))
		proc_count = 5
		AV_master = copy.deepcopy(self.ActionValueTable)
		test = mp.Array('d', AV_master)
		self.ActionValueTable = test
		# conn = [mp.Pipe() for i in range(proc_count)]
		# parent_conn = [c[0] for c in conn]
		# child_conn = [c[1] for c in conn]
		q_p = mp.Queue()
		pdb.set_trace()
		# p_list = [mp.Process(target=self.converge_multi_sub, args=(q_p,steps,epsilon,AV)) for i in range(proc_count)]
		# [p.start() for p in p_list]
		# for i in range(N):
		# 	self.AV_update(updated_VA)
		# 	print(self.ActionValueTable)
		# 	if len(p_list) > 10:
		# 		p_list.pop(0).join(1)
		# 		print('Processes: ')
		# [p.join() for p in p_list]
		pdb.set_trace()


	def converge_multi_sub(self,q,steps,epsilon,AV=None):
		print('We are running!')
		# if AV:
		for s in range(steps):
			a = self.EpsilonGreedyChooseAction(epsilon)
			r = self.Bandit.slots[a].sample()
			self.V_update(a, r)
			# q.put(self.ActionValueTable)
		return self.ActionValueTable

class GridWorldVisualizer(object):
	def __init__(self):
		pygame.init()
		self.BLACK = (0, 0, 0)
		self.WHITE = (255, 255, 255)
		self.GREEN = (0, 255, 0)
		self.RED = (255, 0, 0)
		self.w = 700
		self.h = 500
		self.size = (self.w,self.h)

		self.show = show
		if show:
			self.done = False
			self.screen = pygame.display.set_mode(self.size)
			pygame.display.set_caption("Grid World")
			self.clock = pygame.time.Clock()
			self.startWorld()

	def startWorld(self):
		while not self.done:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.done = True
			self.screen.fill(self.WHITE)
			self.drawWorld()
			pygame.display.flip()
			self.clock.tick(60)
		pygame.quit()

	def drawWorld(self):
		#draws the basic world
		h_lines = np.linspace(0,self.h,self.grid[0]+1)
		# pdb.set_trace()
		for h in h_lines:
			pygame.draw.line(self.screen, self.GREEN, [0, h], [self.w, h], 5)

		v_lines = np.linspace(0,self.w,self.grid[1]+1)
		for v in v_lines:
			pygame.draw.line(self.screen, self.GREEN, [v, 0], [v, self.w], 5)


class GridWorldPlayer(BanditPlayer):
	def __init__(self, world):
		super(GridWorldPlayer, self).__init__(world)
		self.world = world
		self.grid = self.world.grid #hxw
		self.locations = np.array([ [ [x,y] for y in range(self.grid[1])] for x in range(self.grid[0])]).flatten().reshape(-1,2)

	def chooseRandomAction(self):
		a = np.random.randint(self.world.num_actions)
		return a

	def EpsilonGreedyChooseAction(self, epsilon):
		if np.random.rand(1) < epsilon:
			# choose at random
			action = np.random.randint(self.num_actions)
		else:
			# always chooses the first max that it finds in list
			action = np.argmax(self.ActionValueTable)
		return action


class World_1(object):
	def __init__(self):
		self.grid = (5,10)
		self.world = np.zeros(self.grid)
		self.reward = np.ones(self.grid) * -1
		self.reward[1, -1] = 100 #location of door
		self.world[1,-1] = 1
		x,y = self.startLocation()
		self.world[x,y] = 2


		# 0 = No move
		# 1 = Up
		# 2 = Right
		# 3 = Down
		# 4 = Left
		self.num_actions = 5

	def startLocation(self):
		return self.randomLocation()

	def randomLocation(self):
		x = np.random.randint(self.grid[0])
		y = np.random.randint(self.grid[1])
		return (x,y)

	def reward(self, xy):
		# returns the reward at that location
		return self.reward[xy[0], xy[1]]










if __name__ == '__main__':
	# means = [1, 1.5, 2, 2, 1.75]
	# variances = [5, 1, 1, 2, 10]
	# B = Bandit(means, variances)
	# BP = BanditPlayer(B)
	# BP.V_settings(0.2)
	# # BP.converge(10, 0.2, 10000)
	# # BP.plotVATable(BP.ActionValueTable)
	# # R = BP.test_VA(10, 10)
	# # BP.converge(100, 0.2, 10000)
	# # R = BP.test_VA(100, 10)
	# # BP.plotVATable(BP.ActionValueTable)

	# BPMP = BanditPlayerMP(B)
	# BPMP.V_settings(0.2)
	# BPMP.converge_multi(10, 0.2, 100)
	# pdb.set_trace()

	# G = GridWorld((5,10))
	W = World_1()
	pprint([W.reward[(np.random.randint(5), np.random.randint(10))] for i in range(100)])




