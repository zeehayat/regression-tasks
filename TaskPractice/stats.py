import numpy as np
costs = np.array([12.0, 30.0, 8.0, 45.0, 15.0])

# Average Manually

n=len(costs)
mean_manual=np.sum(costs)/n
print(mean_manual)
mean_numpy=np.mean(costs)
print(mean_numpy)

# deviations -> Deviation from the mean

deviations=costs-costs.mean()
print(deviations)

# Sum of Deviations
print(deviations.sum())

variance_manual=np.sum(deviations ** 2)/n
print(variance_manual)

variance_numpy=np.var(costs)
print(variance_numpy)

cable_km=np.array([12.0,30.0,5.0,40.0, 15.0])
x_dev=costs-costs.mean()
y_dev=cable_km-cable_km.mean()

cov_manual=np.sum(x_dev * y_dev)
print(cov_manual)

cov_numpy=np.cov(costs,cable_km,ddof=0)[0,1]
print("Manual Covariance || Numpy Covariance \n",
    cov_numpy, "||\t\t\t", cov_manual)
# standard deviation


# Correlation

def describe(values, label):
    print(f"{label}: mean={values.mean():.2f}, std={values.std():.2f}, "
          f"min={values.min():.2f}, max={values.max():.2f}")

describe(costs, "costs Million PKR")
describe(cable_km, "cable length in km")
print(f"correlation(cost, cable_km) = {np.corrcoef(costs, cable_km)[0,1]:.3f}")