import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

data = pd.read_csv('datasets/SupervisedLearning/k_nearest_neighbours/Iris.csv')

x = data.iloc[:, 1:5]  # iloc -> index location [rows, columns]
y = data.iloc[:, -1]

standardScaler = StandardScaler()

x = standardScaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)

prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

print(accuracy * 100)
print(cm * 100)

# Above code trains the model with a k value of 1
# But 1 is not the optimal K value
# Below code block can be used to find the optimal K value

correctAnswerCounts = []
for i in range(1, 20):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    correct = np.sum(pred == y_test)
    correctAnswerCounts.append(correct)


result = pd.DataFrame(data=correctAnswerCounts)
result.index = result.index + 1
print(result.T)

##select the index of the highest number of accurate results and use it as the k value
