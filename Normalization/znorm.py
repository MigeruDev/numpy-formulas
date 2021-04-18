import numpy as np

'''
    This python code calculates the Z-Normalization

    The absolute value of z represents the distance between that raw score x and 
    the population mean in units of the standard deviation. 
    Z is negative when the raw score is below the mean, positive when above.
    
    z={x-\mu  \over \sigma }

'''

# Input the data array
data = [200, 400, 800, 1000, 2000]

# Putting the data in new numpy array
nparray = np.array(data)

#------------- Normalizing the data --------------------------#
# Obtaining the mean of the array
u = nparray.mean()
# Obtaining the standard deviation of the array
std = nparray.std()

zdata = (nparray - u) / std

print(zdata)