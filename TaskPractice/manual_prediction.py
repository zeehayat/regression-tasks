import numpy as np

X =np.array([
    [15.0, 100.0],
    [30.0, 250.0],
    [5.0, 500.0],
]
)

beta =np.array([0.05,0.002])
predictions = X@beta
print(predictions)

row_vector=[[1.0, 2.0, 3.0]]
col_vector=[[1.0], [2.0], [3.0]]

print(row_vector+col_vector)


y=[4,3,5,6,9,11]

y_hat=[4,3,5,5,9,12]


# SSR =start from 1 uptil n and sum the values of (original-prediction)squared
rst=0.0
for i in range(len(y)):
    rst+=(y[i]-y_hat[i])**2

print (rst)


def squareIt(x):
    return x**2
mse=0
for original, prediction in zip(y,y_hat):
    mse+=(original-prediction) ** 2
print(mse)