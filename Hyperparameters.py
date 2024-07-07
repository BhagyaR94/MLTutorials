import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = pd.read_csv('datasets/DPAndMLTechniques/hyperparameters/Iris.csv')
x = data.drop("Species", axis=1)
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = SVC()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Accuracy with default hyperparameters: ", accuracy_score(prediction, y_test) * 100)
print("\n")

print("GRID SEARCH CV")
param_grid = {'C': [0.1, 1, 10], 'kernel': ['poly', 'linear', 'rbf']}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid)
grid_search.fit(x_train, y_train)
print("Grid Search Best hyperparameters: ", grid_search.best_params_)
print(grid_search.score(x_test, y_test) * 100)

print("\n")
print("RANDOM SEARCH CV")
param_dist = {'C': [0.1, 1, 10], 'kernel': ['poly', 'linear', 'rbf']}
random_vc = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5)
random_vc.fit(x_train, y_train)
print("Random Search Best hyperparameters: ", random_vc.best_params_)
print(random_vc.score(x_test, y_test) * 100)
