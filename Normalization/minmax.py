import numpy as np

'''
    This python code calculates the Min-Max Normalization

    Min-max normalization is one of the most common ways to normalize data. 
    For every feature, the minimum value of that feature gets transformed into a 0, 
    the maximum value gets transformed into a 1, and every other value 
    gets transformed into a decimal between 0 and 1.

    norm = [(x - X.min)/(X.max - X.min)]*(new_max - new_min) + new_min

'''

# Input the data array
data = [200, 400, 800, 1000, 2000]

# Setting the new min and new max
nmin = 0
nmax = 10

# Putting the data in new numpy array
nparray = np.array(data)

#------------- Normalizing the data --------------------------#
# Difference between max nparray value and min nparray value
diff = nparray.max() - nparray.min()
npmin = nparray.min()

ndata = ( (nparray - npmin) / (diff) ) * (nmax-nmin) + nmin

print(ndata)