import numpy as np

weights = {
    'excellent': 3,
    'good': 2,
    'fair': 1    
}

data = np.array([
    'excellent',
    'fair',
    'good',
    'excellent'
])

# Parsing the categories to weights
data = np.fromiter(map(lambda x: weights[x], data), dtype=np.int)

# Creating a zero matrix for the Proximity measure matrix
dmatrix  = np.zeros((data.shape[0], data.shape[0]))

# Get Pmeasure matrix dimensions
x, y = dmatrix.shape

# Max range in the array
M = data.max()

# Looping a lower triangular matrix
for j in range(0, y):
    for i in range(j+1, y):
        # d(i, j) = Z(i) - Z(j)
        # Z(i) = (r(i) - 1) / M - 1
        # Where Z(i) is the norm of "i" and r(i) is the weight of "i"   
        # Carrying out the sum of fractions and simplifying, 
        # the following equation remains
        dmatrix[i,j] = np.abs(data[i] - data[j])/(M - 1)

print(dmatrix)