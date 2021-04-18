import numpy as np

# Input the data Attribute (A.K.A. column, feature, etc.)
attr = np.array([
    'code A',
    'code B',
    'code C',
    'code A'
])

# The fastest way to find occurences within an array
# Check it out in this stackoverflow thread: https://stackoverflow.com/questions/10741346/numpy-most-efficient-frequency-counts-for-unique-values-in-an-array
def unique_count(A):
    A = attr
    unique, inverse = np.unique(A, return_inverse=True)
    count = np.zeros(len(unique), dtype=int)
    np.add.at(count, inverse, 1)
    return np.vstack((unique, count)).T

# Get the occurences in 2D-array
counts = unique_count(attr)

# Number of unique categories of the Attribute (attr)
p = counts[:,0].size

# Parsing the 2D-array to dictionary for convenience
# We subtract one unit by convention from the method used
dcounts = dict(zip(counts[:,0], counts[:,1].astype(int) -1))

# Creating a zero matrix for the Proximity measure matrix
dmatrix  = np.zeros((attr.shape[0], attr.shape[0]))

# Get Pmeasure matrix dimensions
x, y = dmatrix.shape

# Looping a lower triangular matrix
for j in range(0, y):
    for i in range(j+1, x):
        # m (in this case "dcounts") is the number of occurences
        # p is the total of unique categories/variables
        if attr[i] == attr[j]:
            dmatrix[i,j] = (p - dcounts[attr[i]])/p
        else:
            dmatrix[i,j] = 1

print(dmatrix)