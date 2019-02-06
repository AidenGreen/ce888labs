import numpy as np
import matplotlib.pyplot as plt

rowNum = 100
colNum = 5
samples = np.arange(rowNum * colNum).reshape(rowNum,colNum)

for i in range(rowNum):
    for j in range(colNum):
        samples[i][j] = np.random.randint(0,543)
        
print("Max :",np.max(samples))
print("Min :",np.min(samples))
print("Mean :",np.mean(samples))

for i in range(colNum):
    print("Channel:", i)
    print(np.mean(samples[:,i]))


N_points = 1000
n_bins = 2

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
plt.hist()
plt.show()