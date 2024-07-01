import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('./datasets/logistic_regression/people_dataset.csv')

plt.scatter(data.Age, data.Employment_Status)
# plt.show()

x = data[["Age"]]
y = data[["Employment_Status"]]

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2
)

model = LogisticRegression()

model.fit(X_train, y_train)
prediction = model.predict(X_test)

score = model.score(X_test, y_test)

print(score*100)
