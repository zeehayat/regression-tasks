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


