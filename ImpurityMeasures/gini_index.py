import numpy as np
import pandas as pd

'''
     Gini impurity is a measure of how often a randomly 
     chosen element from the set would be incorrectly labeled 
     if it was randomly labeled according to the distribution 
     of labels in the subset. 
'''

# Read example from csv
#example = "ImpurityMeasures/example.csv"
example = "ImpurityMeasures/example2.csv"
data = pd.read_csv(example)

# The fastest way to find occurences within an array
# Check it out in this stackoverflow thread: 
# https://stackoverflow.com/questions/10741346/numpy-most-efficient-frequency-counts-for-unique-values-in-an-array
def unique_count(A):
    unique, inverse = np.unique(A, return_inverse=True)
    count = np.zeros(len(unique), dtype=int)
    np.add.at(count, inverse, 1)
    return np.vstack((unique, count)).T

# Get the gini impurity of one node
def gini(N):
    uniques = unique_count(N)
    total = len(N)
    # Probability of every node leaf Pi
    Pi = np.zeros(uniques.shape[0])
    for i in range(uniques.shape[0]):
        pi = uniques[i,1]/total
        Pi[i] = pi 

    # Appliying the Gini formula: 1 - Sum[Pi(t)^2]
    gini = 1 - np.sum(Pi**2)

    print(uniques)
    print("Probabilities of Pi: ", Pi)
    print("Gini impurity: {}\n".format(gini))
    return gini

# Finding the purest node within data
def findPurest():
    purest = {
        "column": "",
        "gini": 1
    }

    for node in data:
        print("------------ {} ------------".format(node))
        gini_ = gini(data[node].values)

        if gini_ < purest['gini']:
            purest['column'] = node
            purest['gini'] = gini_

    print("The purest node is: {} \nWith an gini index: {}"
    .format(purest['column'], purest['gini']))

    return purest['gini']

# You can select the Father node
#father = findPurest() # is the purest node
father = "Clase"
child = "Talla_Camisa"

def giniWeighted():
    #
    crosstab = pd.crosstab(data[child], data[father], margins=True, margins_name="Total")
    print(crosstab,'\n')
    index = crosstab.index
    crosstab = crosstab.values
    # Sum the ginis
    giniW = 0

    for i in range(len(index)-1):
        print("--------------- {} ---------------".format(index[i]))
        pi = crosstab[i,:-1]/crosstab[i,-1]
        gini_ = 1 - np.sum(pi**2)

        # Calculate the weighted gini and sum
        giniW += (crosstab[i,-1]/crosstab[-1,-1])*gini_

        print("Probabilities of Pi: {}\nGini impurity: {}\n"
        .format(pi, gini_))
    
    print("The weighted gini is: ", giniW)
    return giniW

giniW = giniWeighted()
