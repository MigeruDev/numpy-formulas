import numpy as np

'''
    The angle cosine in geometry can be used to 
    measure the difference between two vector directions. 
    This concept is used in machine learning to measure the 
    difference between sample vectors.

    The cosine of the angle is in the range [-1,1]. The larger the cosine of the angle, 
    the smaller the angle between the two vectors, and the smaller the cosine of the angle 
    indicates the larger the angle between the two vectors. When the directions of the 
    two vectors coincide, the cosine of the angle takes the maximum value of 1. 
    When the directions of the two vectors are completely opposite, 
    the cosine of the angle takes the minimum value of -1.

'''

data = np.array([
    [5, 0, 3, 0, 2, 0, 0, 2, 0, 0], # A
    [3, 0, 2, 0, 1, 1, 0, 1, 0, 1]  # B
])

# Dot product of two arrays
dot = np.dot(data[0,:], data[1,:])
# Module of A
Amod = np.linalg.norm(data[0,:])
# Module of B
Bmod = np.linalg.norm(data[1,:])

# Applying the formula 
cosine =  dot / (Amod*Bmod)

print(cosine)