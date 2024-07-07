import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

data = pd.read_csv('datasets/DPAndMLTechniques/cross_validation/Iris.csv')
x = data.drop("Species", axis=1)
y = data['Species']

knn = KNeighborsClassifier()
svm = SVC()
rfc = RandomForestClassifier()
naiveBayes = GaussianNB()

print("KNN CV = ", cross_val_score(knn, x, y, cv=5).mean() * 100)
print("SVM CV = ", cross_val_score(svm, x, y, cv=5).mean() * 100)
print("RFC CV = ", cross_val_score(rfc, x, y, cv=5).mean() * 100)
print("NB CV = ", cross_val_score(naiveBayes, x, y, cv=5).mean() * 100)
