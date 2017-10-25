import numpy as np
from pprint import pprint
import pdb
import matplotlib.pyplot as plt
import copy
import multiprocessing as mp
import pygame
import copy
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import csv
import time

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

	def saveAVTable(self, fn, AV):
		with open(fn, 'wb') as csvfile:
			writer = csv.writer(csvfile)
			for run in AV:
				writer.writerow(run)

	def loadAVTable(self, fn):
		with open(fn, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			AVTable = next(reader)
		return AVTable

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

		self.done = False
		self.screen = pygame.display.set_mode(self.size)
		pygame.display.set_caption("Grid World")
		self.clock = pygame.time.Clock()

	def setWorld(self, world):
		self.world = world

	def setTrajectories(self, traj_list):
		self.trajs = traj_list

	def startWorld(self):
		while not self.done:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.done = True
			self.screen.fill(self.WHITE)
			self.drawWorld()
			self.drawGoal()
			pygame.display.flip()
			self.clock.tick(60)
		pygame.quit()

	def drawWorld(self):
		#draws the basic world
		h_lines = np.linspace(0,self.h,self.world.grid[0]+1)
		# pdb.set_trace()
		for h in h_lines:
			pygame.draw.line(self.screen, self.GREEN, [0, h], [self.w, h], 5)

		v_lines = np.linspace(0,self.w,self.world.grid[1]+1)
		for v in v_lines:
			pygame.draw.line(self.screen, self.GREEN, [v, 0], [v, self.w], 5)

	def drawGoal(self):
		x = self.h / (self.world.grid[0] + 1) * self.world.goal[0]
		y = self.w / (self.world.grid[1] + 1) * self.world.goal[1]
		pygame.draw.rect(self.screen, self.RED, (x, y, 20, 20))

class GridWorldPlayer(BanditPlayer):
	def __init__(self, world):
		super(GridWorldPlayer, self).__init__(world)
		self.world = world
		self.grid = self.world.grid #hxw

	def chooseRandomAction(self):
		a = np.random.randint(self.world.num_actions)
		return a

	def EpsilonGreedyChooseAction(self, epsilon, table=None):
		if table is None:
			table = self.ActionValueTable
		if np.random.rand(1) < epsilon:
			# choose at random
			action = np.random.randint(self.num_actions)
		else:
			# always chooses the first max that it finds in list
			action = np.argmax(table)
		return action

	def testEGreedy(self, epsilon):
		pprint([self.EpsilonGreedyChooseAction(epsilon) for i in range(100)])

	def plotTrajectory(self, traj, ax):
		for i_t in range(len(traj)-1):
			xs = [traj[i_t][0], traj[i_t+1][0]]
			ys = [traj[i_t][1], traj[i_t+1][1]]
			ax.plot(xs,ys,'b-')

	def plotTrajectoryHeatMap(self, trajs, ax):
		heatmap = np.zeros((self.world.grid[0], self.world.grid[1]))
		for traj in trajs:
			for x,y in traj:
				heatmap[x,y] += 1
		# pdb.set_trace()
		# heatmap /= np.max(heatmap)
		# light is zero
		ax.imshow(heatmap.T, cmap='binary', interpolation='none', origin='lower')
		self.plotPatch(self.world.goal, ax)

	def HeatMapTest(self):
		# just to check what is high value and low
		# dark = high
		# light = low
		ax = plt.subplot(1,1,1)
		in_ones = np.ones((10,10))
		in_ones[4,4] = 0
		ax.imshow(in_ones, cmap='binary', interpolation='none', origin='lower')
		plt.show()

	def plotPatch(self, goal, ax):
		s = 1.0
		rect = mpatches.Rectangle([goal[0]-s/2, goal[1]-s/2], 1, 1, fill=False, linewidth=1, edgecolor='r')
		ax.add_patch(rect)
		return rect

	def plotTrajSim(self, traj, ax):
		plt.ion()
		plt.show()
		last_step = [0,0]
		for i,step in enumerate(traj):
			p_cur = self.plotPatch(step, ax)
			txt = ax.text(0,0,'%s: [%s,%s]' %(i,step[0],step[1]))
			txt2 = ax.text(0,1, '%s: [%s,%s]' %(i-1,last_step[0],last_step[1]))
			plt.draw()
			plt.pause(0.5)
			p_cur.remove()
			txt.remove()
			txt2.remove()
			last_step = step



	# def plotLearning(self, traj_all, ax):
	# 	for traj in traj_all:


	def saveTraj(self, traj_all, fn):
		with open(fn, 'wb') as csvfile:
			writer = csv.writer(csvfile)
			for traj in traj_all:
				writer.writerow(np.array(traj))

	def loadTraj(self, fn):
		traj_all = []
		with open(fn, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				traj = []
				for r in row:
					x,y = np.array(r.strip('[]').split()).astype(int)
					traj.append(np.array([x,y]))
				np.array(traj)
				traj_all.append(copy.deepcopy(traj))
		return traj_all
	
	# def plotTrajectory(self, traj, ax): # DO THIS FIRST!

class GridWorldPlayerAV(GridWorldPlayer):
	def __init__(self, world):
		super(GridWorldPlayerAV, self).__init__(world)

	def V_update(self, action, r):
		self.ActionValueTable[action] = self.ActionValueTable[action] + self.alpha * (r - self.ActionValueTable[action])

	def initalizeZero(self):
		self.ActionValueTable = np.zeros(self.num_actions)

	def set_AV_Table(self, AV):
		self.ActionValueTable = AV

	def initializeRandom(self):
		self.ActionValueTable = np.random.rand(self.num_actions)

	def initializeOptimistic(self, opt_val = 1e1):
		self.ActionValueTable = np.zeros(self.num_actions) + opt_val

	def converge(self, steps, epsilon, N):
		if self.alpha is None:
			raise(ValueError('Must set alpha value first with V_settings'))
		traj_all = []
		for i in range(N):
			traj = []
			pos = self.world.startLocation()
			traj.append(copy.deepcopy(pos))
			for s in range(steps):
				a = self.EpsilonGreedyChooseAction(epsilon)
				pos = self.world.updatePosition(a)
				r = self.world.getReward(pos)
				self.V_update(a, r)
				traj.append(copy.deepcopy(pos))
			traj_all.append(copy.deepcopy(traj))
		return traj_all

	def plotVATable(self, VA, show=True):
		width = 1.0
		ax = plt.subplot(1,1,1)
		f = ax.get_figure()
		ind = np.arange(self.num_actions)
		ax.bar(ind, VA, width, color='r', label='ValueAction')
		# ax.bar(ind+width, self.Bandit.means, color='b', yerr=self.Bandit.variances)
		plt.legend()
		if show: plt.show()
		return ax

	def saveAVTable(self, fn, AV):
		with open(fn, 'wb') as csvfile:
			writer = csv.writer(csvfile)
			for run in AV:
				writer.writerow(run)

	def loadAVTable(self, fn):
		with open(fn, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			AVTable = next(reader)
		return AVTable

class GridWorldPlayerQ(GridWorldPlayer):
	def __init__(self, world):
		super(GridWorldPlayerQ, self).__init__(world)
		self.QTable = np.ones((self.grid[0], self.grid[1], self.world.num_actions)) * 1.0
		self.discount = None

	def Q_settings(self, alpha, discount):
		self.discount = discount
		self.alpha = alpha

	def QTable_update(self, action, pos, next_pos, r, update = True):
		# updates that value in the Q table
		val = self.QTable[pos[0], pos[1], action]
		# add value of best action from next state
		max_next = max(self.QTable[next_pos[0], next_pos[1]])
		val = val + self.alpha * (r + self.discount*max_next - val)
		if update: self.QTable[pos[0], pos[1], action] = val
		return val

	def convergeQtable(self, steps, epsilon, N):
		if self.alpha is None:
			raise(ValueError('Must set alpha value first with Q_settings'))
		if self.discount is None:
			raise(ValueError('Must set discount value first with Q_settings'))
		traj_all = []
		for i in range(N):
			traj = []
			pos = self.world.startLocation()
			# pos = np.array([8,1])
			self.world.pos = pos
			traj.append(pos)
			for s in range(steps):
				prev_pos = copy.deepcopy(pos)
				a = self.EpsilonGreedyChooseAction(epsilon, table=self.QTable[self.world.pos[0], self.world.pos[1]])
				# a = 2
				pos = self.world.updatePosition(a)
				r = self.world.getReward(pos)
				self.QTable_update(a, prev_pos, pos, r)
				traj.append(copy.deepcopy(pos))
				# pdb.set_trace()
			traj_all.append(traj)
		return traj_all

	def testQTable(self, steps, N):
		epsilon = 0 # no exploration when testing
		traj_all = []
		for i in range(N):
			traj = []
			pos = self.world.startLocation()
			self.world.pos = pos
			traj.append(pos)
			for s in range(steps):
				prev_pos = copy.deepcopy(pos)
				a = self.EpsilonGreedyChooseAction(epsilon, table=self.QTable[self.world.pos[0], self.world.pos[1]])
				pos = self.world.updatePosition(a)
				r = self.world.getReward(pos)
				traj.append(copy.deepcopy(pos))
			traj_all.append(traj)
		return traj_all

	def plotQTable_Heatmap(self, ax):
		heatmap = np.zeros((self.world.grid[0], self.world.grid[1]))
		for x in range(self.world.grid[0]):
			for y in range(self.world.grid[1]):
				heatmap[x,y] = max(self.QTable[x,y])
		ax.imshow(heatmap.T, cmap='binary', interpolation='none', origin='lower')
		self.plotPatch(self.world.goal, ax)


	def plotQTable_Quiver(self, table, ax):
		arrow_scale = 1.0
		f = ax.get_figure()
		X,Y = np.meshgrid(np.arange(self.grid[0]), np.arange(self.grid[1]))
		for a in range(5):
			U = np.zeros_like(X)
			V = np.zeros_like(Y)
			for ix,x in enumerate(range(self.grid[0])):
				for iy,y in enumerate(range(self.grid[1])):
					Q_total = np.sum(abs(self.QTable[x,y]))
					# div = (max(self.QTable[x,y]) - min(self.QTable[x,y]))
					# if div == 0:
					# 	div = 1
					# Q = (self.QTable[x,y,a] - min(self.QTable[x,y])) / div
					# Q *= arrow_scale
					# a_best = np.argmax(self.QTable[x,y])
					# u,v = self.world.action_list[a_best]
					u,v = self.world.action_list[a]
					try:
						# U[iy,ix] = Q*u/Q_total
						# V[iy,ix] = Q*v/Q_total
						U[iy,ix] = Q*u
						V[iy,ix] = Q*v
					except:
						print("%s, %s" %(ix,iy))
			ax.quiver(X,Y,U,V)
		r = 0.25
		return ax

	def plotQTable_Quiver(self, table, ax):
		f = ax.get_figure()
		X,Y = np.meshgrid(np.arange(self.grid[0]), np.arange(self.grid[1]))
		U = np.zeros_like(X)
		V = np.zeros_like(Y)
		for ix,x in enumerate(range(self.grid[0])):
			for iy,y in enumerate(range(self.grid[1])):
				a = np.argmax(self.QTable[x,y])
				# Q_total = np.sum(abs(self.QTable[x,y]))
				# div = (max(self.QTable[x,y]) - min(self.QTable[x,y]))
				# if div == 0:
				# 	div = 1
				# Q = (self.QTable[x,y,a] - min(self.QTable[x,y])) / div
				# Q *= arrow_scale
				# a_best = np.argmax(self.QTable[x,y])
				# u,v = self.world.action_list[a_best]
				u,v = self.world.action_list[a]
				try:
					# U[iy,ix] = Q*u/Q_total
					# V[iy,ix] = Q*v/Q_total
					U[iy,ix] = u
					V[iy,ix] = v
				except:
					print("%s, %s" %(ix,iy))
		ax.quiver(X,Y,U,V)
		self.plotPatch(self.world.goal, ax)
		return ax

	def saveQTable(self, table, fn):
		np.save(fn, table) #adds .npy to the end!

	def loadQTable(self, fn):
		table = np.load(fn+'.npy')
		return table


class GridWOrldPlayerQ_Multi(GridWorldPlayerQ):
	def __init__(self, world):
		super(GridWOrldPlayerQ_Multi, self).__init__(world)
		self.procs = 8
		pipes = [mp.Pipe() for proc in range(self.procs)]
		self.parent_conn  = [pipe[0] for pipe in pipes]
		self.child_conn  = [pipe[1] for pipe in pipes]
		self.counter = 0

	def startMP_test(self):
		self.P = []
		for proc in range(self.procs):
			self.P.append(mp.Process(target=self.testMP, args=(proc, self.child_conn[proc])))
			self.P[-1].start()

	def sendMessages(self, msg):
		for proc in range(self.procs):
			self.parent_conn[proc].send(msg)

	def endMP(self):
		for P in self.P:
			if P.is_alive():
				P.join(2)
				print('Process Ended')

	def testMP(self, proc, child_conn):
		print('Process Started: %s' %proc)
		keep_running = True
		while keep_running:
			if child_conn.poll():
				msg = child_conn.recv()
				print('Message Received: %s' %msg)
				self.counter += 1
				print('Counter: %s' %self.counter)
				if msg == -1:
					keep_running = False

	def startMP(self):
		self.P = []
		for proc in range(self.procs):
			self.P.append(mp.Process(target=self.testMP, args=(proc, self.child_conn[proc])))
			self.P[-1].start()

	def singleEpisodeLoop(self, steps, epsilon, child_conn):
		keep_running = True
		while keep_running:
			if child_conn.poll():
				msg = child_conn.recv()
				# print('Message Received: %s' %msg)
				if isinstance(msg, int) and msg == -1:
					keep_running = False
					continue
				# start iteration loop here
				self.QTable = copy.deepcopy(msg) # setting local object table
				traj = []
				pos = self.world.startLocation()
				traj.append(pos)
				for s in range(steps):
					prev_pos = copy.deepcopy(pos)
					a = self.EpsilonGreedyChooseAction(epsilon, table=self.QTable[self.world.pos[0], self.world.pos[1]])
					pos = self.world.updatePosition(a)
					r = self.world.getReward(pos)
					self.QTable_update(a, prev_pos, pos, r, update = True)
					traj.append(copy.deepcopy(pos))
				child_conn.send((self.QTable, traj))
		print('Processes Ended')

	def allProcessesAlive(self):
		print('Checking if Processes are Alive')
		all_alive = False
		while not all_alive:
			all_alive = all([p.is_alive() for p in self.P])
			time.sleep(0.1)
			print('Current Status: %s' %[p.is_alive() for p in self.P])
		print('All Processes Started!')

	def sendEndMessage(self):
		[parent_conn.send(-1) for parent_conn in self.parent_conn]
		print('End Messages Sent')

	def convergeQtable(self, steps, epsilon, N):
		self.ASync_alpha = 0.5
		if self.alpha is None:
			raise(ValueError('Must set alpha value first with Q_settings'))
		if self.discount is None:
			raise(ValueError('Must set discount value first with Q_settings'))
		# start processes
		self.P = []
		for proc in range(self.procs):
			self.P.append(mp.Process(target=self.singleEpisodeLoop, args=(steps,epsilon,self.child_conn[proc])))
			self.P[-1].start()
		self.allProcessesAlive()

		traj_all = []
		# add Q table to each one
		[self.parent_conn[proc].send(self.QTable) for proc in range(self.procs)]
		counter = 0
		while counter < N:
			for pc in self.parent_conn:
				if pc.poll():
					updatedQTable, traj = pc.recv()
					# i am making an update step that is like the Q learning update
					self.QTable = self.QTable + self.ASync_alpha * (updatedQTable - self.QTable)
					pc.send(copy.deepcopy(self.QTable)) #let it start on the next iteration
					traj_all.append(traj)
					counter += 1
				if counter > N:
					break
		self.sendEndMessage()
		self.endMP()
		return traj_all


class World_1(object):
	def __init__(self):
		self.grid = (10,5)
		self.goal = (9,1)
		self.world = np.zeros(self.grid)
		self.reward = np.ones(self.grid) * -1
		self.reward[self.goal[0], self.goal[1]] = 100 #location of door
		self.world[-2,-1] = 1
		self.pos = self.startLocation()
		self.world[self.pos[0],self.pos[1]] = 2
		# 0 = No move
		# 1 = Up
		# 2 = Right
		# 3 = Down
		# 4 = Left
		self.num_actions = 5
		self.action_list = {0:[0,0], 1:[0,1], 2:[1,0], 3:[0,-1], 4:[-1,0]}
		self.locations = np.array([ [ [x,y] for y in range(self.grid[1])] for x in range(self.grid[0])]).flatten().reshape(-1,2)

	def startLocation(self):
		self.pos = self.randomLocation()
		return self.pos

	def randomLocation(self):
		x = np.random.randint(self.grid[0])
		y = np.random.randint(self.grid[1])
		return np.array([x,y])

	def getReward(self, xy):
		# returns the reward at that location
		r = self.reward[xy[0], xy[1]]
		return r

	def test(self):
		pprint([self.reward[(np.random.randint(self.grid[0]), np.random.randint(self.grid[1]))] for i in range(100)])

	def getNextPosition(self, a, xy):
		pos = copy.deepcopy(xy)
		move = self.action_list[a]
		pos += move
		pos = self.clampToWorld(pos)
		return pos

	def updatePosition(self, a):
		pos = self.getNextPosition(a, self.pos)
		self.pos = pos
		return pos

	def clampToWorld(self,xy):
		xy[0] = np.clip(xy[0], 0, self.grid[0]-1).astype(int)
		xy[1] = np.clip(xy[1], 0, self.grid[1]-1).astype(int)
		return xy

	def inWorld(self,xy):
		xy_fix = self.clampToWorld(xy)
		if (xy == xy_fix).all():
			return False
		else:
			return True

	def neighbors(self, xy):
		nbrs = []
		for a in range(self.num_actions):
			nbrs.append(self.getNextPosition(a, xy))
		nbrs = np.unique(nbrs, axis = 0)
		return nbrs

	def test_neighbors(self):
		for i in range(100):
			xy = self.randomLocation()
			print('Start Location: %s, Neighbors: %s' %(xy,self.neighbors(xy)))



class PlotsForHomework(object):
	def CollectDataBandit(self):
		# collect all the data needed to produce plots
		learning_rate = 0.2
		means = [1, 1.5, 2, 2, 1.75]
		variances = [5, 1, 1, 2, 10]
		B = Bandit(means, variances)
		BP = BanditPlayer(B)
		BP.V_settings(learning_rate)
		stat_runs = 10
		for iterations, steps in ((1e5, 10),(1e4, 100)): #same number of steps
			for epsilon in (0.0, 0.2):
				AV_tables = []
				for stat in range(stat_runs):
					BP.initalizeOptimistic(10)
					BP.converge(steps, epsilon, int(iterations))
					AV_tables.append(BP.ActionValueTable)
				fn = ('Data/bandit_a%0.1f_e%0.1f_c%s' %(learning_rate,epsilon,int(iterations))).replace('.','D')+'.csv'
				BP.saveAVTable(fn, AV_tables)
				print('Completed -- Iterations: %s, Steps: %s, Epsilon: %s' %(iterations, steps, epsilon))

	def CollectAVLearningData(self):
		W = World_1()
		learning_rate = 0.2
		steps = 20
		epsilon = 0.2
		steps = 20
		iterations = 1e4
		## AV Player ##
		GW = GridWorldPlayerAV(W)
		GW.V_settings(learning_rate)
		stat_runs = 10
		fn = ('Data/gridAV_a%0.1f_e%0.1f_c%s' %(learning_rate,epsilon,int(iterations))).replace('.','D')+'.csv'
		AV_tables = []
		for stat in range(stat_runs):
			GW.initializeOptimistic(2)
			traj_all = GW.converge(steps, epsilon, int(iterations))
			AV_tables.append(GW.ActionValueTable)
			GW.saveTraj(traj_all, fn.replace('.csv', '_traj%s.csv' %stat))
			print('Completed Stat Run %s' %stat)
		GW.saveAVTable(fn, AV_tables)


	def CollectQLearningData(self):
		W = World_1()
		## Q Multi Learner Player ##
		GWM = GridWOrldPlayerQ_Multi(W)
		alpha = 0.2
		discount = 0.9
		epsilon = 0.2
		N = int(1e4)
		steps = 20
		stat_runs = 10
		test_runs = 20
		GWM.Q_settings(alpha = alpha, discount = discount)
		# ax = plt.subplot(1,1,1)
		fn = ('Data/QLearner_a%0.1f_d%0.1f_e%0.1f_c%s' %(alpha,discount,epsilon,N)).replace('.','D')+'.csv'
		for stat in range(stat_runs):
			traj_all = GWM.convergeQtable(steps, epsilon, N)
			traj_all_test = GWM.testQTable(steps, 100)
			GWM.saveTraj(traj_all,  fn.replace('.csv', '_traj%s.csv' %stat))
			GWM.saveQTable(GWM.QTable, fn.replace('.csv', '_Q%s' %stat))



		# GWM.plotTrajectoryHeatMap(traj_all[-100:], ax)
		# GWM.plotTrajectoryHeatMap(traj_all_test, ax)
		# traj_all = GWM.loadTraj(fn)
		# GWM.plotTrajectoryHeatMap(traj_all[-100:], ax)
		# GWM.plotTrajSim(traj_all_test[-1], ax)
		# plt.show()



	def PlotBandit(self):
		# bar graph that compares both learners to the true value
		BP.plotVATable(BP.ActionValueTable)
		R = BP.test_VA(10, 10)
		BP.converge(100, 0.2, 10000)
		R = BP.test_VA(100, 10)
		BP.plotVATable(BP.ActionValueTable)

		BPMP = BanditPlayerMP(B)
		BPMP.V_settings(0.2)
		BPMP.converge_multi(10, 0.2, 100)
		pdb.set_trace()

	# def PlotGridAV(self):
		# GW.plotVATable(GW.ActionValueTable)
		# pprint(GW.ActionValueTable)
		# ax = plt.subplot(1,1,1)
		# traj_all = GW.convergeQtable(20, 0.1, 1000)







if __name__ == '__main__':
	PFH = PlotsForHomework()
	PFH.CollectDataBandit()
	PFH.CollectAVLearningData()
	PFH.CollectQLearningData()

	## Q Learner Player ##
	# GW = GridWorldPlayerQ(W)
	# alpha = 0.2
	# discount = 0.9
	# epsilon = 0.0
	# N = 1000
	# ax = plt.subplot(1,1,1)
	# GW.Q_settings(alpha = alpha, discount = discount)
	# traj_all = GW.convergeQtable(20, epsilon, N)
	# traj_all_test = GW.testQTable(20, 100)
	# GW.plotQTable_Heatmap(ax)
	# fn = ('QLearner_a%0.1f_d%0.1f_e%0.1f' %(alpha,discount,epsilon)).replace('.','D')+'.csv'
	# GW.saveTraj(traj_all, fn)
	# traj_all2 = GW.loadTraj(fn)
	# GW.plotQTable_Quiver(GW.QTable, ax)
	# GW.plotTrajectoryHeatMap(traj_all[-100:], ax)
	# GW.plotTrajectoryHeatMap(traj_all_test, ax)
	# GW.plotQTable(GW.QTable, ax)
	# plt.show()
	# pdb.set_trace()