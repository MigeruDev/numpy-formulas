import numpy as np
import pandas as pd

'''
    
'''

# Input the data
data = np.array([
    ["Jack", 1, 0, 1, 0, 0, 0],
    ["Mary", 1, 0, 1, 0, 1, 0],
    ["Jim", 1, 1, 0, 0, 0, 0]
])

patients = data.shape[0]

for i in range(patients-1):
    for j in range(i+1, patients):
        # Getting the contingency matrix
        # This method is slow, can be improved
        contingency_matrix = pd.crosstab(data[i,1:], data[j,1:])

        r = contingency_matrix["0"]["1"]
        s = contingency_matrix["1"]["0"]
        q = contingency_matrix["1"]["1"]

        d = (r + s)/(q + r + s)

        print("d({}, {})\t=\t{}".format(data[i,0], data[j,0], d))