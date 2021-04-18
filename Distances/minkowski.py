import numpy as np

'''
    Minkowski distance is a distance/ similarity measurement between two points 
    in the normed vector space (N dimensional real space) and is a generalization of 
    the Euclidean distance and the Manhattan distance.
'''

objA = [22, 1, 42, 10]

objB = [20, 0, 36, 8]

h = 3

npA = np.array(objA)

npB = np.array(objB)

minkowski = (np.abs(npA - npB) ** h).sum() ** (1/h)

print(minkowski)