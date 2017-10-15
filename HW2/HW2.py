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
from collections import defaultdict
from multiprocessing import Process
ITER_LIMIT = 100000
REPEAT = 10
ITERATIONS = int(1e4)

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


	def plotSolution(self, sol, ax = None, show = True):
		if ax is None: f,ax = plt.subplots(1,1)
		self.plotPoints(show = False, ax = ax)
		sol_list = self.get_sol_list(sol)
		for c1,c2 in sol_list:
			x1,y1 = self.cities[c1]
			x2,y2 = self.cities[c2]
			ax.arrow(x1,y1, x2-x1, y2-y1, head_width = 20)
		if show: plt.show()
		return ax

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
		sols = []
		for i in range(N):
			s_neighbor = self.neighbor_sol(s)
			s_score = self.evaluate(s)
			s_neighbor_score = self.evaluate(s_neighbor)
			s_list = ((s,s_score), (s_neighbor, s_neighbor_score))
			# print('Current Temp: %s,' %((N-i)/N))
			s_choosen = self.accept(s_list, (N-i), N)
			s = s_choosen[0]
			sols.append(copy.deepcopy(s))
			scores.append(s_choosen[1])
		print('Solution: %s' %s)
		return scores, sols

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
		sol_list_all = []
		for i in range(N):
			noise = 1.0 - i/N
			parent_list = self.choose_parents(sol_list, Mutations, noise)
			child_list = [self.perturb_sol(p[0], noise) for p in parent_list]
			child_list.extend(sol_list)
			sol_list = self.accept(child_list, Pop_total)
			scores.append(np.sort([s[1] for s in sol_list]))
			# sol_list_all.append(s[0] for s in np.sort(sol_list, key=lambda x:x[1]))
		# return scores, sol_list_all
		return scores, min(sol_list, key=lambda x: x[1])

	def printSortedValues(self, sols):
		pprint(np.sort([p[1] for p in sols]))

class MonteCarloTreeSearch(SimulatedAnnealing):
	def __init__(self, distanceMatrix, epsilon = 1.0):
		super(MonteCarloTreeSearch, self).__init__(distanceMatrix)
		self.tree = dict.fromkeys(range(self.sol_length))
		for k in self.tree.keys(): #setting initial value
			self.tree[k] = self.createBlankNode([k])
		self.epsilon = epsilon
		self.best_distance = 1e9
		self.best_solution = []

	def createBlankNode(self, path):
		node = dict.fromkeys(['path', 'value', 'visits', 'children', 'parent'])
		node['path'] = path
		node['value'] = 0
		node['visits'] = 0
		if len(node['path']) == self.sol_length: #applies stopping condition
			node['children'] = None
		else:
			node['children'] = []
		return node

	def Guess(self, exclude = None):
		#returns a random choice to start tree
		if exclude is None:
			first = np.random.choice(range(self.sol_length))
		else:
			remaining = set(range(self.sol_length)) - set(exclude)
			first = np.random.choice(list(remaining))
		return first

	def addNode(self, parent, next_exploration=None):
		# randomly select future states
		# make sure node doesn't exist
		if next_exploration is None:
			next_exploration = self.Guess(exclude = parent['path'])
		path = parent['path'] + [next_exploration]
		if len(path) != len(set(path)):
				pdb.set_trace()
		child = self.createBlankNode(path)
		child['visits'] += 1
		# playout scenario
		play_out = self.Playout(path)
		child['value'] = self.evaluate(play_out)
		parent['children'].append({next_exploration:child})

		# backprop value up chain
		self.updateCosts(child['path'])
		
		# update tracker if best score
		if self.best_distance > child['value']:
			self.best_distance = child['value']
			self.best_solution = play_out

	def findLeaf(self, parent):
		if parent['children'] is None:
			# a solution can't be any longer
			return
		if parent['children'] == []:
			#found a leaf
			self.addNode(parent)
		else:
			# epsilon greedily choose next step
			if np.random.rand(1) < self.epsilon: # do random choice
				child_indx =  self.Guess(exclude = parent['path'])
				all_children = [p.keys()[0] for p in parent['children']]
				# if state exists, go there
				if child_indx in all_children:
					child_node = self.get_Child_Node_From_Node(parent, child_indx)
					self.findLeaf(child_node)
				else:
				# if state doesn't exist, then create it
					self.addNode(parent, child_indx)
			else:
				child_node = self.get_Min_Child_From_Node(parent)
				self.findLeaf(child_node)

	def get_Min_Child_From_Node(self, parent):
		child_scores = list()
		for c in parent['children']:
			child_scores.append([c.keys()[0],c[c.keys()[0]]['value']])
		# add something if values are equal
		___, min_score = min(child_scores, key=lambda x:x[1])
		min_indxs = [k[0] for k in child_scores if k[1] == min_score]
		child_indx = np.random.choice(min_indxs)
		child_node = self.get_Child_Node_From_Node(parent,child_indx)
		return child_node

	def get_Child_Node_From_Node(self, node, indx):
		# given a node, it will search the children and return the node that corresponds
		# to the child
		all_children = [C.keys()[0] for C in node['children']]
		ix = all_children.index(indx)
		desired_node = node['children'][ix][indx]
		return desired_node

	def traverseTree(self, path):
		node = self.tree[path[0]] #deals with head of tree
		try:
			if len(path) > 1:
				for p in path[1:]:
					node = self.get_Child_Node_From_Node(node, p)
		except:
			pprint("Path: %s" %path)
			raise AttributeError("'Node does not exist'")
		return node

	def updateCosts(self, path):
		while len(path) >= 2:
			current_node = self.traverseTree(path)
			parent_node = self.traverseTree(path[:-1])
			parent_node['value'] = (current_node['value'] + parent_node['value'] * parent_node['visits'])/(parent_node['visits'] + 1)
			parent_node['visits'] += 1
			path = path[:-1]

	def Playout(self, path):
		play_out = copy.deepcopy(path)
		while len(play_out) < self.sol_length:
			play_out.append(self.Guess(exclude = play_out))
		return play_out

	def rootTreeChoice(self):
		#root is structured slightly differently so need a different method
		# same idea though
		if np.random.rand(1) < self.epsilon:
			child_indx = np.random.choice(range(self.sol_length))
		else:
			child_scores = [[k,self.tree[k]['value']] for k in self.tree.keys()]
			min_score = min([[k,self.tree[k]['value']] for k in self.tree.keys()], key=lambda x: x[1])[1]
			min_indx = [k for k in self.tree.keys() if self.tree[k]['value'] == min_score]
			child_indx = np.random.choice(min_indx)
		return child_indx

	def findSolution(self, N):
		scores = []
		sol_list = []
		for i in range(N):
			first = self.rootTreeChoice()
			self.findLeaf(self.tree[first])
			self.epsilon = float(N-i)/N
			scores.append(self.best_distance)
			sol_list.append(self.best_solution)
		return scores, sol_list

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
		print('Simulated Annealing Solution')
		scores_all = []
		times_all = []
		N = ITERATIONS
		for fn,sc in zip(self.fn_list, self.scenarios):
			scores = []
			sols = []
			times = []
			for i in range(REPEAT):
				F = fileReader(fn)
				start_time = time.time()
				SA = SimulatedAnnealing(F.distanceMatrix)
				times.append(time.time() - start_time)
				score,sol = SA.findSolution(N)
				scores.append(score)
				sols.append(sol)
			scores_all.append(scores)
			times_all.append(times)
			with open('SA_Results/%scities_%s.csv' %(sc,'score'),'wb') as csvfile:
				writer = csv.writer(csvfile)
				[writer.writerow(s) for s in np.array(scores).T]
			for i,s_i in enumerate(sols):
				with open('SA_Results/%scities_%s_%s.csv' %(sc,'sols',i),'wb') as csvfile:
					writer = csv.writer(csvfile)
					[writer.writerow(s) for s in s_i]
			with open('SA_Results/%scities_%s.csv' %(sc,'time'),'wb') as csvfile:
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
		P = 10
		M = 5
		N = ITERATIONS/(P)
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
		print('Solving With Evolutionary Algorithm')
		P = 10
		M = 5
		N = int(ITERATIONS/M)
		for fn,sc in zip(self.fn_list, self.scenarios):
			scores = []
			max_scores = []
			times = []
			sols = []
			for i in range(REPEAT):
				F = fileReader(fn)
				EA = EvolutionaryAlgorithm(F.distanceMatrix)
				start_time = time.time()
				s,sol = EA.findSolution(Pop_total = P, Mutations = M, N = N)
				times.append(time.time() - start_time)
				scores.append(s)
				max_scores.append([max(all_s) for all_s in s])
				sols.append(sol)

				for s_i in scores:
					with open('EA_Results/%scities_%s_%s.csv' %(sc,'scores',i),'wb') as csvfile:
						writer = csv.writer(csvfile)
						[writer.writerow(s) for s in np.array(s_i)]
			with open('EA_Results/%scities_%s.csv' %(sc,'score'),'wb') as csvfile:
				writer = csv.writer(csvfile)
				[writer.writerow(s) for s in np.array(max_scores).T]

			with open('EA_Results/%scities_%s.csv' %(sc,'sols'),'wb') as csvfile:
				if '..' in csvfile.name:
					pdb.set_trace()
				writer = csv.writer(csvfile)
				for s in np.array(sols):
					writer.writerow(np.append(s[0], s[1]))
			with open('EA_Results/%scities_%s.csv' %(sc,'time'),'wb') as csvfile:
				writer = csv.writer(csvfile)
				[writer.writerow([s]) for s in np.array(times).T]

	def MCTSSoltuions_multiple(self):
		print('Solving with MCTS')
		N = ITERATIONS
		for fn,sc in zip(self.fn_list, self.scenarios):
			scores = []
			times = []
			sols = []
			for i in range(REPEAT):
				F = fileReader(fn)
				MCTS = MonteCarloTreeSearch(F.distanceMatrix)
				start_time = time.time()
				s, sol = MCTS.findSolution(N = N)
				times.append(time.time() - start_time)
				scores.append(s)
				sols.append(sol)
			with open('MCTS_Results/%scities_%s.csv' %(sc,'score'),'wb') as csvfile:
				writer = csv.writer(csvfile)
				[writer.writerow(s) for s in np.array(scores).T]
			for i,s_i in enumerate(sols):
				with open('MCTS_Results/%scities_%s_%s.csv' %(sc,'sols',i),'wb') as csvfile:
					writer = csv.writer(csvfile)
					[writer.writerow(s) for s in s_i]
			with open('MCTS_Results/%scities_%s.csv' %(sc,'time'),'wb') as csvfile:
				writer = csv.writer(csvfile)
				[writer.writerow([t]) for t in times]

	def CitiesResults(self):
		for fn_prefix,data_file in zip(self.scenarios,self.fn_list):
			# prints out results from 15 cities test for each algorithm
			folders = ['%s_Results/' %i for i in ('SA','EA','MCTS')]
			# folders = ['%s_Results/' %i for i in ('SA','MCTS')]
			time_fn = '%scities_time.csv' %fn_prefix
			times_all = []
			for folder in folders:
				with open(folder + time_fn, 'rb') as csvfile:
					reader = csv.reader(csvfile)
					time_scenario = [float(row[0]) for row in reader]
					print(folder)
					print('Mean:%sf, StDev:%sf' %(np.mean(time_scenario), np.std(time_scenario)))
				times_all.append(time_scenario)

			##### City Graphs #####
			solution_fn = '%scities_score.csv' %fn_prefix
			f1,ax1 = plt.subplots(1,1)
			n = int(1e2)
			formats = ['rx', 'bo', 'g*']
			for folder,fmt in zip(folders, formats):
				scores = []
				with open(folder + solution_fn, 'rb') as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:
						scores.append([float(r) for r in row])
					print(folder)
					print('Final Score: %0.2f, StDev: %0.2f' %(np.mean(scores[-1]), np.std(scores[-1])))
					x_range = np.arange(len(scores))[::n]
					if 'EA' in folder:
						x_range *= 5
						
					ax1.errorbar(x_range, np.mean(scores, axis = 1)[::n], yerr = np.std(scores, axis = 1)[::n], label = folder.split('_')[0], fmt = fmt)
					i = 1
			ax1.legend()
			ax1.set_title('%s Cities Solution Search Progress' %fn_prefix)
			ax1.set_xlabel('Iterations')
			ax1.set_ylabel('Solution Distance')
			ax1.set_xlim([0, ITERATIONS])
			ax1.get_figure().savefig('%sCity_Solutions.png' %fn_prefix)
			plt.close(ax1.get_figure())
			# plt.show()

			for folder in folders:
				sol_file = [folder+'%scities_sols_%s.csv' %(fn_prefix,i) for i in range(REPEAT)]
				F = fileReader(data_file)
				GS = GenerateSolution(F.distanceMatrix)
				sols = []
				score = 1e10
				if 'EA' in folder:
					with open(folder + '%scities_sols.csv' %fn_prefix, 'rb') as csvfile:
						reader = csv.reader(csvfile)
						for row in reader:
							if float(row[-1]) < score:
								sols = copy.deepcopy(np.array(row[:-1]).astype(float).astype(int))
								score = float(row[-1])
				else:
					for f in sol_file:
						with open(f, 'rb') as csvfile:
							reader = csv.reader(csvfile)
							row = np.array(list(reader)[-1]).astype(int)
						if GS.evaluate(row) < score:
							sols = copy.deepcopy(row)
							score = GS.evaluate(row)
				f2,ax2 = plt.subplots(1,1)
				F.plotSolution(sols, ax = ax2, show = False)
				if 'SA' in folder:
					name = 'Simulated Annealing'
				elif 'EA' in folder:
					name = 'Evolutionary Algorithm'
				else:
					name = 'MCTS'
				ax2.set_title('%s Cities Best Solution for %s - %0.2f' %(fn_prefix, name, score))
				ax2.set_xlabel('X Position')
				ax2.set_ylabel('Y Position')
				ax2.get_figure().savefig('%sCity_%s.png' %(fn_prefix,folder.split('_')[0]))





		pdb.set_trace()


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

	# MCTS = MonteCarloTreeSearch(F.distanceMatrix)
	# for __ in np.arange(1e5):
	# 	first = MCTS.rootTreeChoice()
	# 	MCTS.findLeaf(MCTS.tree[first])
	# pprint('Best Solution:%s' %(MCTS.best_solution))
	# pprint('Min Distance:%s' %(MCTS.best_distance))

	# pdb.set_trace()
	PFH = PlotsForHomework()

	# p = []
	# p.append(Process(target=PFH.SASolutions_multiple))
	# p.append(Process(target=PFH.EASolutions_multiple))
	# p.append(Process(target=PFH.MCTSSoltuions_multiple))
	# [P.start() for P in p]
	# [P.join() for P in p]


	PFH.CitiesResults()

