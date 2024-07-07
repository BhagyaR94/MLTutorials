import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('datasets/SupervisedLearning/decision_tree/kyphosis.csv')
x = data.drop('Kyphosis', axis=1)
y = data['Kyphosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=50)  #n_estimators = number of decision trees
model.fit(x_train, y_train)

prediction = model.predict(x_test)
print(prediction)
print("\n")
print(y_test)

print(accuracy_score(prediction, y_test) * 100)
