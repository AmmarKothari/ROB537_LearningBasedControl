import numpy as np
from pprint import pprint
import pdb
import matplotlib.pyplot as plt




class SlotMachine(object):
	# defines a single slot machine in N armed bandit problem
	# Gaussian distribution of rewards
	def __init__(self, mean, variance, dist = 'Gaussian'):
		self.mean = float(mean)
		self.variance = float(variance)
		self.dist = dist

	def sample(self, n=1):
		if self.dist.lower() == 'Gaussian'.lower():
			out = np.random.normal(loc=self.mean, scale=self.variance, size=n)
		if self.dist.lower() == 'Uniform'.lower():
			out = np.random.uniform(low=self.mean-self.variance, high=self.mean+self.variance, size=n)
		out[out<0] = 0 #set all negative values to zero
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
			pprint('Slot Output:%s' %np.array([s.sample() for s in self.slots]).flatten())


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


	def converge(self, steps, epsilon, N):
		# rewrite this to be asynchronous updates!
		# create processes
		# evertime you get something back, update parameters and spawn a new process
		if self.alpha is None:
			raise(ValueError('Must set alpha value first with V_settings'))
		for i in range(N):
			for s in range(steps):
				a = self.EpsilonGreedyChooseAction(epsilon)
				r = self.Bandit.slots[a].sample()
				self.V_update(a, r)

	def test_VA(self, steps, N):
		epsilon = 0 # or should i keep it the same??
		rs = []
		for i in range(N):
			r = 0
			for s in range(steps):
				a = self.EpsilonGreedyChooseAction(epsilon)
				r =+ self.Bandit.slots[a].sample()
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





if __name__ == '__main__':
	means = [1, 1.5, 2, 2, 1.75]
	variances = [5, 1, 1, 2, 10]
	B = Bandit(means, variances)
	BP = BanditPlayer(B)
	BP.V_settings(0.2)
	BP.converge(10, 0.2, 10000)
	BP.plotVATable(BP.ActionValueTable)
	R = BP.test_VA(10, 10)
	BP.converge(100, 0.2, 10000)
	R = BP.test_VA(100, 10)
	BP.plotVATable(BP.ActionValueTable)
	pdb.set_trace()




