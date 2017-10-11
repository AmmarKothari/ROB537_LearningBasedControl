import csv
import pdb
from pprint import pprint
import numpy as np
import numpy.matlib
import scipy.spatial
from itertools import cycle
import matplotlib.pyplot as plt
import copy

ITER_LIMIT = 100000

class fileReader(object):
	def __init__(self, fn):
		self.fn = fn
		self.loadFile(fn)
		self.distanceMatrix()

	def loadFile(self, fn):
		#loads the file and contents
		with open(fn, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			self.num_city = int(reader.next()[0])
			self.cities = []
			for row in reader:
				self.cities.append([float(r) for r in row])

	def plotPoints(self, show = True, ax = None):
		if ax is None:
			f,ax = plt.subplots(1,1)
		ax.plot(np.array(self.cities)[:,0],np.array(self.cities)[:,1], 'rx')
		if show: plt.show()
		return ax

	def distanceMatrix(self):
		# self.distanceMatrix = np.matlib.zeros((self.num_city, self.num_city))
		# # pdb.set_trace()
		# for i1,c1 in enumerate(self.cities):
		# 	x1,y1 = c1[:]
		# 	pprint('X1:%s, Y1:%s' %(x1,y1))
		# 	for i2,c2 in enumerate(self.cities):
		# 		x2,y2 = c2[:]
		# 		self.distanceMatrix[i1,i2] = np.linalg.norm(np.array(c1) - np.array(c2))

		self.distanceMatrix = scipy.spatial.distance.cdist(self.cities, self.cities)

	def get_sol_list(self, sol):
		if isinstance(sol, list):
			sol_list = zip(sol, sol.append(sol.pop(0)))
		else:
			sol_list = zip(sol, np.roll(sol,-1))
		return sol_list


	def plotSolution(self, sol):
		f,ax = plt.subplots(1,1)
		self.plotPoints(show = False, ax = ax)
		sol_list = self.get_sol_list(sol)
		for c1,c2 in sol_list:
			x1,y1 = self.cities[c1]
			x2,y2 = self.cities[c2]
			ax.arrow(x1,y1, x2-x1, y2-y1, head_width = 20)
		plt.show()

	# def plotMultipleSolutions(self, sol):

class GenerateSolution(object):
	def __init__(self, distanceMatrix):
		self.distanceMatrix = distanceMatrix
		self.sol_length = distanceMatrix.shape[0]

	def initialGuess(self):
		sol = np.random.permutation(self.sol_length)
		return sol

	def generator(self, n, limit = ITER_LIMIT):
		# n is the number of solutions
		for iL in range(limit):
			sol = [np.random.permutation(self.sol_length) for iN in range(n)]
			yield sol

	def get_sol_list(self, sol):
		if isinstance(sol, list):
			sol_list = zip(sol, sol.append(sol.pop(0)))
		else:
			sol_list = zip(sol, np.roll(sol,-1))
		return sol_list

	def evaluate(self, sol):
		sol_list = self.get_sol_list(sol)
		cost = 0
		for c1,c2 in sol_list:
			cost += self.distanceMatrix[c1,c2]
		return cost


class SimulatedAnnealing(GenerateSolution):
	def __init__(self, distanceMatrix):
		super(SimulatedAnnealing, self).__init__(distanceMatrix)

	def generator(self, n, limit = ITER_LIMIT):
		for iL in range(limit):
			sol = [np.random.permutation(self.sol_length) for iN in range(n)]
			yield sol

	def neighbor_sol(self, orig_sol):
		#randomly switch two successive states
		sol = copy.deepcopy(orig_sol)
		i1 = np.random.choice(len(sol))
		#whether to switch value in front or behind
		if np.random.choice(2):
			i2 = i1 + 1
		else:
			i2 = i1 - 1

		if i2 >= len(sol):
			i2 = 0
		elif i2<0:
			i2 = len(sol)-1
		sol[i1],sol[i2] = sol[i2],sol[i1]
		return sol

	def accept(self, sol_list, T, T_max):
		# select solution based on Temperature
		# scale T from 0 to 1
		if T > T_MAX:
			print('Max Temp is 100')
			return False
		if T < 0:
			print('Min Temp is 0')
			return False
		if np.random.rand(1) <= T/T_MAX:
			sol_indx = np.random.choice(len(sol_list))
			sol = sol_list[sol_indx]
		else:
			sol = max(sol_list, key=lambda x: x[1])
		return sol

	def findSolution(self, N = 100):
		s = self.initialGuess()
		scores = []
		for i in range(N):
			s_neighbor = self.neighbor_sol(s)
			s_score = O.evaluate(s)
			s_neighbor_score = O.evaluate(s_neighbor)
			s_list = ((s,s_score), (s_neighbor, s_neighbor_score))
			# print('Current Temp: %s,' %((N-i)/N))
			s_choosen = O.accept(s_list, (N-i)/N)
			s = s_choosen[0]
			scores.append(s_choosen[1])
		print('Solution: %s' %s)
		return scores

	def plotSearch(self, scores):
		plt.plot(scores)
		plt.x_label('Iterations')
		plt.y_label('')
















class PlotsForHomework(object):
	def __init__(self):
		fn_list = ['hw2.data/%scities.csv' %(15,25,100)]
		fn_list.append('hw2.data/25cities_A.csv')


	def plotOriginalData(self):
		for fn in fn_list:
			F = fileReader(fn)
			ax = F.plotSolution(sol, show = False)
			pdb.set_trace()
			ax.getfig.savefig(fn.replace('hw2.data/', ''))







if __name__ == '__main__':
	F = fileReader('hw2.data/15cities.csv')
	# G = GenerateSolution(F.distanceMatrix)
	# g = G.generator(1)
	# for i in range(10):
	# 	sol = next(g)[0]
	# 	G.evaluate(sol)
	# 	F.plotSolution(sol)

	O = SimulatedAnnealing(F.distanceMatrix)
	O.findSolution()

	# s = O.initialGuess()
	# # for i in range(1000):
	# s_new = O.neighbor_sol(s)
	# print('Solution: %s' %s)
	# print('Neighbor: %s' %s_new)
	# s_score = O.evaluate(s)
	# s_new_score = O.evaluate(s_new)
	# print('Current Solution: %0.2f' %s_score)
	# print('New Solution: %0.2f' %s_new_score)
	# s_list = ((s,s_score), (s_new, s_new_score))
	# s_new = O.accept(s_list, 100)
	# print('Chosen Solution: %0.2f' %s_new[1])

