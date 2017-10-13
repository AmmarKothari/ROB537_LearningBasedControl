import csv
import pdb
from pprint import pprint
import numpy as np
import numpy.matlib
import scipy.spatial
from itertools import cycle
import matplotlib.pyplot as plt
import copy
import time

ITER_LIMIT = 100000
REPEAT = 10

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
		ax.set_xlabel('X Location')
		ax.set_ylabel('Y Location')
		ax.set_yticks(np.arange(0,1001,200))
		ax.set_xticks(np.arange(0,2001,500))
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
		# creates a set of tuples where each row is the start and end city
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
			sol_shift = copy.deepcopy(sol)
			sol_shift.append(sol_shift.pop(0))
			sol_list = zip(sol, sol_shift)
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

	# def generator(self, n, limit = ITER_LIMIT):
	# 	for iL in range(limit):
	# 		sol = [np.random.permutation(self.sol_length) for iN in range(n)]
	# 		yield sol

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
		T = float(T)
		if T > T_max:
			print('Max Temp is 100')
			return False
		if T < 0:
			print('Min Temp is 0')
			return False
		if np.random.rand(1) <= T/T_max:
			sol_indx = np.random.choice(len(sol_list))
			sol = sol_list[sol_indx]
		else:
			sol = min(sol_list, key=lambda x: x[1])
		return sol

	def findSolution(self, N = 100):
		s = self.initialGuess()
		scores = []
		for i in range(N):
			s_neighbor = self.neighbor_sol(s)
			s_score = self.evaluate(s)
			s_neighbor_score = self.evaluate(s_neighbor)
			s_list = ((s,s_score), (s_neighbor, s_neighbor_score))
			# print('Current Temp: %s,' %((N-i)/N))
			s_choosen = self.accept(s_list, (N-i), N)
			s = s_choosen[0]
			scores.append(s_choosen[1])
		print('Solution: %s' %s)
		return scores

	def plotSearch(self, scores):
		f,ax = plt.subplots(1,1)
		ax.plot(scores)
		ax.x_label('Iterations')
		ax.y_label('Total Distance')

class EvolutionaryAlgorithm(SimulatedAnnealing):
	def __init__(self, distanceMatrix):
		super(EvolutionaryAlgorithm, self).__init__(distanceMatrix)

	# def neighbor_sol(self, orig_sol):
	# 	raise AttributeError("'This method is not available for this class'")

	def choose_parents(self, sol_list, N, noise):
		# returns the N solutions to perturb
		# noise is between 0 and 1
		parent_list = []
		for __ in range(N):
			if np.random.rand(1) < noise:
				indx = np.random.choice(len(sol_list))
				parent_list.append(sol_list[indx])
			else:
				parent_list.append(min(sol_list, key=lambda x: x[1]))
		return parent_list

	def perturb_sol(self, orig_sol, noise):
		# randomly switches state based on noise (0 to 1)
		if noise > 1 or noise < 0:
			raise AttributeError("'Invalid noise value. Must be between 0 and 1'")
		#should this be probabilitically set??
		switch_count = int(np.round(len(orig_sol) * noise, 0))
		new_sol = [sol for sol in orig_sol]
		for i in range(switch_count):
			new_sol = self.neighbor_sol(new_sol)
		cost = self.evaluate(new_sol)
		return (new_sol, cost)

	def accept(self, sol_list, N):
		# remove the (total_solutions - N) worst solutions
		accept_list = sorted(sol_list, key=lambda x:x[1])[:N]
		return accept_list

	def findSolution(self, Pop_total = 10, Mutations = 5, N = 100):
		rand_sol_gen = self.generator(Pop_total)
		sol_list = [(s,self.evaluate(s)) for s in next(rand_sol_gen)]
		scores = []
		for i in range(N):
			noise = 1.0 - i/N
			parent_list = self.choose_parents(sol_list, Mutations, noise)
			child_list = [self.perturb_sol(p[0], noise) for p in parent_list]
			child_list.extend(sol_list)
			sol_list = self.accept(child_list, Pop_total)
			scores.append(np.sort([s[1] for s in sol_list]))
		return scores, min(sol_list, key=lambda x: x[1])

	def printSortedValues(self, sols):
		pprint(np.sort([p[1] for p in sols]))





class PlotsForHomework(object):
	def __init__(self):
		self.fn_list = ['hw2.data/%scities.csv' %(i) for i in (15,25,100)]
		self.fn_list.append('hw2.data/25cities_A.csv')
		self.scenarios = ['15', '25', '100', '25A']


	def plotOriginalData(self):
		for fn in self.fn_list:
			F = fileReader(fn)
			ax = F.plotPoints(show = False)
			ax.set_title('%s Cities' %F.num_city)
			if 'A' in fn:
				ax.set_title('%sA Cities' %F.num_city)
			ax.get_figure().savefig(fn.replace('hw2.data/', '').replace('csv', 'png'))

	def plotSASolutions(self):
		scores = []
		N = 10000
		for fn in self.fn_list:
			F = fileReader(fn)
			SA = SimulatedAnnealing(F.distanceMatrix)
			scores.append(SA.findSolution(N))
		f,ax = plt.subplots(1,1)
		[ax.plot(s, label = n) for s,n in zip(scores,self.scenarios)]
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Distance')
		ax.legend()
		ax.get_figure().savefig('SimulatedAnnealing_Single.png')

	def SASolutions_multiple(self):
		scores_all = []
		times_all = []
		N = 10000
		for fn in self.fn_list:
			scores = []
			times = []
			for i in range(REPEAT):
				F = fileReader(fn)
				start_time = time.time()
				SA = SimulatedAnnealing(F.distanceMatrix)
				times.append(time.time() - start_time)
				scores.append(SA.findSolution(N))
			scores_all.append(scores)
			times_all.append(times)
			with open(fn.replace('hw2.data/','SA_Results/'),'wb') as csvfile:
				writer = csv.writer(csvfile)
				[writer.writerow(s) for s in np.array(scores).T]
			with open(fn.replace('hw2.data/','SA_Results/').replace('.','_time.'),'wb') as csvfile:
				writer = csv.writer(csvfile)
				[writer.writerow([s]) for s in np.array(times).T]
		# avg = [np.mean(s,axis = 0) for s in scores_all]
		# stdev = [np.std(s,axis = 0) for s in scores_all]
		# avg_time = [np.mean(t,axis=0) for t in times_all]
		# stdev_time = [np.mean(t,axis=0) for t in times_all]

		# f,ax = plt.subplots(1,1)
		# n = 10
		# [ax.errorbar(np.arange(0,N,n),a[::n], yerr = s[::n], label=l) for a,s,l in zip(avg,stdev,self.scenarios)]
		# ax.set_xlabel('Iterations')
		# ax.set_ylabel('Distance')
		# ax.legend()
		# ax.set_title('Simulated Annealing')
		# ax.get_figure().savefig('SimulatedAnnealing_Stat.png')

	def plotSASolutions_multiple(self):
		fn_list = ['SA_Results/' + f.split('/')[1] in self.fn_list]
		scores_all = []
		times_all = []
		for fn in fn_list:
			scores = []
			times = []
			with open(fn, 'rb') as csvfile:
				reader = csv.reader(csvfile)
				for row in reader:
					scores.append(row)

	def EASolution(self):
		N = 100
		P = 10
		M = 5
		scores = []
		for fn in self.fn_list:
			F = fileReader(fn)
			EA = EvolutionaryAlgorithm(F.distanceMatrix)
			s,sol = EA.findSolution(Pop_total = P, Mutations = M, N = N)
			scores.append(s)
			f,ax = plt.subplots(1,1)
			ax.set_title(fn)
			[ax.plot(range(N),np.mean(s, axis = 1), label = n) for s,n in zip(scores,self.scenarios)]
			plt.show()

	def EASolutions_multiple(self):
		N = 100
		P = 10
		M = 5
		for fn in self.fn_list:
			scores = []
			times = []
			sols = []
			for i in range(REPEAT):
				F = fileReader(fn)
				EA = EvolutionaryAlgorithm(F.distanceMatrix)
				start_time = time.time()
				s,sol = EA.findSolution(Pop_total = P, Mutations = M, N = N)
				times.append(time.time() - start_time)
				scores.append(s)
				sols.append(sol)

				for s_i in scores:
					with open(fn.replace('hw2.data/','EA_Results/').replace('.','_%s.' %i),'wb') as csvfile:
						writer = csv.writer(csvfile)
						[writer.writerow(s) for s in np.array(s_i)]
			with open(fn.replace('hw2.data/','EA_Results/').replace('.','_time.'),'wb') as csvfile:
				writer = csv.writer(csvfile)
				[writer.writerow([s]) for s in np.array(times).T]

if __name__ == '__main__':
	F = fileReader('hw2.data/15cities.csv')
	# G = GenerateSolution(F.distanceMatrix)
	# g = G.generator(1)
	# for i in range(10):
	# 	sol = next(g)[0]
	# 	G.evaluate(sol)
	# 	F.plotSolution(sol)

	# O = SimulatedAnnealing(F.distanceMatrix)
	# O.findSolution()

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

	PFH = PlotsForHomework()
	# PFH.plotOriginalData()
	# PFH.SASolutions_multiple()
	PFH.EASolutions_multiple()

