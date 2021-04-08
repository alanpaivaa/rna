import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate linear space
x = np.linspace(0, 3, 1000)
y = -x+3
plt.plot(x, y)

# Generate points for class 0
x0 = np.array([
    np.random.uniform(low=1.5, high=2.5, size=10),
    np.random.uniform(low=0, high=1, size=10),
    np.random.uniform(low=0, high=0.8, size=10)
]).ravel()

y0 = np.array([
    np.random.uniform(low=0, high=0.6, size=10),
    np.random.uniform(low=0, high=0.8, size=10),
    np.random.uniform(low=1.5, high=2.2, size=10)
]).ravel()

klass_0 = np.array([x0, y0]).T
klass_0 = np.append(klass_0, np.zeros((klass_0.shape[0], 1)), axis=1)
plt.scatter(klass_0[:, 0], klass_0[:, 1])

# Generate points for class 1
x1 = np.random.uniform(low=2, high=3, size=10)
y1 = np.random.uniform(low=2, high=3, size=10)

klass_1 = np.array([x1, y1]).T
klass_1 = np.append(klass_1, np.ones((klass_1.shape[0], 1)), axis=1)
plt.scatter(klass_1[:, 0], klass_1[:, 1])

# Generate dataset
dataset = np.concatenate((klass_0, klass_1), axis=0)
print(dataset)

# Show plot
plt.show()

# Write dataset to csv file
df = pd.DataFrame(dataset)
df.to_csv('datasets/artificial.csv', index=False, header=False)