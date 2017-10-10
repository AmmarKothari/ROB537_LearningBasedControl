import csv
import pdb
from pprint import pprint
import numpy as np
import numpy.matlib

class fileReader(object):
	def __init__(self, fn):
		self.fn = fn
		self.loadFile(fn)
		self.distanceMatrix()

	def loadFile(self, fn):
		#loads the file and contents
		with open(fn, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			self.num_city = float(reader.next()[0])
			self.cities = []
			for row in reader:
				self.cities.append([float(r) for r in row])

	def distanceMatrix(self):
		self.distanceMatrix = np.matlib.zeros((self.num_city, self.num_city))
		pdb.set_trace()
		for c in self.cities:
			x,y = c[:]
			pprint('X:%s, Y:%s' %(x,y))





if __name__ == '__main__':
	F = fileReader('hw2.data/15cities.csv')
