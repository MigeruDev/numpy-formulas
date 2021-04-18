import numpy as np

'''
    From the name you can guess the calculation of this distance. 
    Imagine that you are driving from an intersection to another intersection in Manhattan. 
    Is the driving distance a straight line between two points? Obviously not, 
    unless you can cross the building. The actual driving distance is this "Manhattan distance" (L1 norm). 
    This is also the source of the Manhattan distance name, which is also known as the City Block distance
'''

objA = [22, 1, 42, 10]

objB = [20, 0, 36, 8]

npA = np.array(objA)

npB = np.array(objB)

manhattan = np.sum(np.abs(npA - npB))

# manhattan = np.linalg.norm(npA - npB, ord=1)

print(manhattan)
















