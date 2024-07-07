import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('./datasets/support_vector_machine/Iris.csv')
data.drop("Id", axis=1)

sb.pairplot(data, hue="Species")

x = data.drop("Species", axis=1)
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = SVC()
model.fit(x_train, y_train)

prediction = model.predict(x_test)
print(accuracy_score(prediction, y_test) * 100)
