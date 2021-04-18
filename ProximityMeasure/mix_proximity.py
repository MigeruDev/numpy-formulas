import numpy as np

# Index of Columns attribute type
ctype = {
    'nominal': [0],
    'ordinal': [1],
    'binary': [],
    'numeric': [2]
}
# Weights for the ordinal attributes
weights = {
    'excellent': 3,
    'good': 2,
    'fair': 1    
}

data = np.array([
    ["code A", "excellent", 45],
    ["code B", "fair", 22],
    ["code C", "good", 64],
    ["code A", "excellent", 28]
])

factor = 1

# ---------------------------------------- NOMINAL ----------------------------------------------------#

# The fastest way to find occurences within an array
# Check it out in this stackoverflow thread: 
# https://stackoverflow.com/questions/10741346/numpy-most-efficient-frequency-counts-for-unique-values-in-an-array
def unique_count(A):
    unique, inverse = np.unique(A, return_inverse=True)
    count = np.zeros(len(unique), dtype=int)
    np.add.at(count, inverse, 1)
    return np.vstack((unique, count)).T

def nom_distance(A):
    # Get the occurences in 2D-array
    counts = unique_count(A)

    # Number of unique categories of the Attribute (attr)
    p = counts[:,0].size

    # Parsing the 2D-array to dictionary for convenience
    # We subtract one unit by convention from the method used
    dcounts = dict(zip(counts[:,0], counts[:,1].astype(int) -1))

    # Creating a zero matrix for the Proximity measure matrix
    nom_matrix  = np.zeros((A.shape[0], A.shape[0]))

    # Get Pmeasure matrix dimensions
    x, y = nom_matrix.shape

    # Looping a lower triangular matrix
    for j in range(0, y):
        for i in range(j+1, x):
            # m (in this case "dcounts") is the number of occurences
            # p is the total of unique categories/variables
            if A[i] == A[j]:
                nom_matrix[i,j] = (p - dcounts[A[i]])/p
            else:
                nom_matrix[i,j] = 1
    
    return nom_matrix

# ---------------------------------------- ORDINAL ----------------------------------------------------#
def ord_distance(A):
    # Parsing the categories to weights
    A = np.fromiter(map(lambda x: weights[x], A), dtype=np.int)

    # Creating a zero matrix for the Proximity measure matrix
    ord_matrix  = np.zeros((A.shape[0], A.shape[0]))

    # Get Pmeasure matrix dimensions
    x, y = ord_matrix.shape

    # Max range in the array
    M = A.max()

    # Looping a lower triangular matrix
    for j in range(0, y):
        for i in range(j+1, y):
            # d(i, j) = Z(i) - Z(j)
            # Z(i) = (r(i) - 1) / M - 1
            # Where Z(i) is the norm of "i" and r(i) is the weight of "i"   
            # Carrying out the sum of fractions and simplifying, 
            # the following equation remains
            ord_matrix[i,j] = np.abs(A[i] - A[j])/(M - 1)

    return ord_matrix

# ---------------------------------------- NUMERIC ----------------------------------------------------#

def num_distance(A):
    A = A.astype(int)

    Amax, Amin = A.max(), A.min()

    # Creating a zero matrix for the Proximity measure matrix
    num_matrix  = np.zeros((A.shape[0], A.shape[0]))

    x, y = num_matrix.shape

    # Looping a lower triangular matrix
    for j in range(0, y):
        for i in range(j+1, x):

            num_matrix[i,j] = np.abs(A[j] - A[i])/(Amax - Amin)
            
    return num_matrix
# ---------------------------------------------------------------------------------#
# To store all proximity matrix
dmatrix = []
# Iterating column types to hash the attribute distance
for attr, values in ctype.items():
    if attr == "nominal":
        for A in values:
            dmatrix.append(nom_distance(data[:,A]))

    elif attr == "ordinal":
        for A in values:
            dmatrix.append(ord_distance(data[:,A]))
            
    elif attr == "numeric":
        for A in values:
            dmatrix.append(num_distance(data[:,A]))

# ----------------------------- MIX PROXIMITY ----------------------------------#    
def mix_proximity(dmatrix):

    x, y = dmatrix[0].shape

    # Creating a zero matrix for the Proximity measure matrix
    mix_matrix  = np.zeros((x, y))

    size = len(dmatrix)

    # Looping a lower triangular matrix
    for j in range(0, y):
        for i in range(j+1, x):
            # Using numpy broadcast to applying the formula
            mix_matrix[i,j] = (dmatrix[:,i,j]*factor).sum()/size 

    return mix_matrix

dmatrix = np.array(dmatrix)

mix_matrix = mix_proximity(dmatrix)

print(mix_matrix)

