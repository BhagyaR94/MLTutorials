import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = np.asarray([[26, 50000], [29, 70000], [34, 55000], [31, 41000]])

minMaxScaler = MinMaxScaler()
minMaxResult = minMaxScaler.fit_transform(data)
print(minMaxResult)

standardScaler = StandardScaler()
standardResult = standardScaler.fit_transform(data)
print(standardResult)
