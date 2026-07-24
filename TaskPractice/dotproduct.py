import numpy as np

features=np.array([15.0,100.0])
weights=np.array([0.05,0.002])

#manual dot

result=features[0]*weights[0]+features[1]*weights[1]

print(result)

loop_total=0.0
for f, w in zip(features, weights):
    loop_total+=f*w
print(loop_total)

dot_total=features @ weights
print(dot_total)

dt_total=np.dot(features,weights)
print(dt_total)