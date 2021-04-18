import numpy as np

'''
    This python code calculates the Min-Max Normalization

    Min-max normalization is one of the most common ways to normalize data. 
    For every feature, the minimum value of that feature gets transformed into a 0, 
    the maximum value gets transformed into a 1, and every other value 
    gets transformed into a decimal between 0 and 1.

    norm = [(x - X.min)/(X.max - X.min)]*(new_max - new_min) + new_min

'''

def minmax_norm(X, npmin, diff, nmax, nmin):
    return ( (X - npmin) / (diff) ) * (nmax-nmin) + nmin

# Input the data array
data = [ 13, 15, 16, 16, 19, 20, 20, 21, 22, 22,
         25, 25, 25, 25, 30, 33, 33, 35, 35, 35,
         35, 36, 40, 45, 46, 52, 70]

# Setting the new min and new max
nmin = 0
nmax = 1

# Putting the data in new numpy array
nparray = np.array(data)

#------------- Normalizing the data --------------------------#
# Difference between max nparray value and min nparray value
diff = nparray.max() - nparray.min()
npmin = nparray.min()

ndata = minmax_norm(nparray, npmin, diff, nmax, nmin)

print(ndata)

# Getting the norm of 35
nvalue = minmax_norm(35, npmin, diff, nmax, nmin)
print(np.round(nvalue, 2))
