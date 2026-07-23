import numpy as np


capacities_kw=np.array([100.0,250.0,500.0])

capacities_watts=capacities_kw*1000.0
print(capacities_kw)
print(capacities_watts)

a= np.array([[1,2,3,4,5,6]])
print(a.shape)

b= np.array([[1,2,3,4,5,6],
[3,4,9,11,17,9]
             ])
print(b.shape)

print ("NEXT LINE===================================================================")
X=np.array([
    [15.0, 100.0, 3],
    [30.0, 250.0, 4],
    [5.0,  500.0, 1]
])

print(X)
print(X.shape)
print("First Row in the array X[0]", X[0]) # First Row in the array
print("First Column All rows X[:,0]", X[:,0])  # First Column All rows
print("row 0 column 1 X[0,1]",X[0,1]) #row 0 column 1
print("first two rows, first two columns X[:2,:2]", X[:2,:2]) #first two rows, first two columns