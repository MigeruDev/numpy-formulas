import numpy as np
import pandas as pd

'''
     entropy helps us to build an appropriate decision tree 
     for selecting the best splitter. Entropy can be defined 
     as a measure of the purity of the sub split. 
     Entropy always lies between 0 to 1. 
'''

# Read example from csv
example = "ImpurityMeasures/example3.csv"
data = pd.read_csv(example)

# The fastest way to find occurences within an array
# Check it out in this stackoverflow thread: 
# https://stackoverflow.com/questions/10741346/numpy-most-efficient-frequency-counts-for-unique-values-in-an-array
def unique_count(A):
    unique, inverse = np.unique(A, return_inverse=True)
    count = np.zeros(len(unique), dtype=int)
    np.add.at(count, inverse, 1)
    return np.vstack((unique, count)).T

# Get the entropy of one node
def entropy(Node):
    uniques = unique_count(data[Node].values)
    total = len(data[Node].values)

    # Probability of every node leaf Pi
    Pi = np.zeros(uniques.shape[0])
    for i in range(uniques.shape[0]):
        pi = uniques[i,1]/total
        Pi[i] = pi 

    # Appliying the Entropy formula: - Sum[Pi(t)*log2(Pi(t))]
    entropy = - np.sum(Pi*np.log2(Pi, out=np.zeros_like(Pi), where=(Pi!=0)))

    print(uniques)
    print("Probabilities of Pi: ", Pi)
    print("Entropy impurity: {}\n".format(entropy))
    return entropy

# [WARNING] This example doesnt support continuous attributes
# if you want to calculate the IG of continuos attributes
# please go to the entropy_c.py file in this same folder
father = "Etiqueta_Clase"
child = "a1"

# Get the entropy of the Father node
father_E = entropy('Etiqueta_Clase')

def informationGain():
    #
    crosstab = pd.crosstab(data[child], data[father], margins=True, margins_name="Total")
    print(crosstab,'\n')
    index = crosstab.index
    crosstab = crosstab.values
    # Sum the entropys
    entropyW = 0

    for i in range(len(index)-1):
        print("--------------- {} ---------------".format(index[i]))
        pi = crosstab[i,:-1]/crosstab[i,-1]
        entropy_ = - np.sum(pi*np.log2(pi, out=np.zeros_like(pi), where=(pi!=0)))

        # Calculate the weighted entropy and sum
        entropyW += (crosstab[i,-1]/crosstab[-1,-1])*entropy_

        print("Probabilities of Pi: {}\nentropy impurity: {}\n"
        .format(pi, entropy_))
    
    print("The information gain is: ", father_E - entropyW)
    return father_E - entropyW

infoGain = informationGain()


