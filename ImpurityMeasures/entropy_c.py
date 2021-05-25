import numpy as np
from numpy.core.numeric import zeros_like
import pandas as pd

# [TODO] This code was made in a hurry. 
# It can be improved, someday I will. Please excuse me

data = {
    "a3": [1.0, 6.0, 5.0, 4.0, 7.0, 3.0,8.0,7.0,5.0],
    "class": ["CP", "CP", "CN", "CP", "CN", "CN", "CN", "CP", "CN"]
}

division = np.array([2.0, 3.5, 4.5, 5.5, 6.5, 7.5])

df = pd.DataFrame(data)

df.sort_values(by=["a3"], inplace=True)

print(df)

E_father = 0.9911

for i in division:
    print("------------------------------------------------------")
    print("Split in ", str(i),"\n")
    dfi = df.copy()
    dfi["a3"] = dfi["a3"].apply(lambda x: "C0" if x <= i else "C1")
    confusion = pd.crosstab(dfi["a3"], dfi["class"], margins=True, margins_name="Total")
    print(confusion)
    index = confusion.index
    confusion = confusion.values

    a = confusion[0,0]/confusion[0,-1]
    b = confusion[0,1]/confusion[0,-1]
    E0 = -(a*np.log2(a, out=np.zeros_like(a), where=(a!=0))) - (b*np.log2(b, out=np.zeros_like(b), where=(b!=0)))
    print("\nEntropy of {}:\t\t{}".format(index[0],E0))
    
    c = confusion[1,0]/confusion[1,-1]
    d = confusion[1,1]/confusion[1,-1]
    E1 = -(c*np.log2(c, out=np.zeros_like(c), where=(c!=0))) - (d*np.log2(d, out=np.zeros_like(d), where=(d!=0)))
    print("Entropy of {}:\t\t{}".format(index[1],E1))

    C0 = confusion[0,-1]/confusion[-1,-1]
    C1 = confusion[1,-1]/confusion[-1,-1]
    InfGain = E_father - ((C0*E0)+(C1*E1))
    print("Information Gain:\t{}".format(InfGain))

