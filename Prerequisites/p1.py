import random
import numpy
from math import sqrt
import matplotlib.pyplot as plt

def findmean(l1):
	return sum(l1)/len(l1)

v1 = [random.randrange(0,1000)]
for i in range(1,100):
	v1 += [int(random.randrange(0,1000))]

v1.sort()
print("vector => " + str(v1))

v1 = v1*3

print("vector => " + str(v1))

print("mean of vector => " + str(findmean(v1)))
print("standard deviation of vector => " + str(numpy.std(v1)))

#question 11
plt.plot(v1)
plt.show()

#question 12
v2 = [i*i for i in v1]
plt.plot(v2)
plt.show()
