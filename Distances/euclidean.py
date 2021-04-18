import numpy as np

'''
    The Euclidean distance (L2 norm) is the easiest to 
    understand distance calculation method derived from 
    the distance formula between two points in Euclidean space
'''

objA = [22, 1, 42, 10]

objB = [20, 0, 36, 8]

npA = np.array(objA)

npB = np.array(objB)

euclidean = np.sqrt(np.sum(np.square(npA - npB)))

# euclidean = np.linalg.norm(npA - npB)

print(euclidean)

