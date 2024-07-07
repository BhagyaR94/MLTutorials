import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plot
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('./datasets/decision_tree/kyphosis.csv')
x = data.drop('Kyphosis', axis=1)
y = data['Kyphosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

sb.pairplot(data, hue="Kyphosis")
# plot.show()

model = GaussianNB()
model.fit(x_train, y_train)

prediction = model.predict(x_test)
print(prediction)
print("\n")
print(y_test)

print(accuracy_score(prediction, y_test) * 100)
