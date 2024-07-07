import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  LabelBinarizer
data = pd.read_csv('datasets/DPAndMLTechniques/one_hot_encoding/student_data.csv')

result_category = data['results']

encoder = LabelEncoder()
result = encoder.fit_transform(result_category)

binerizer = LabelBinarizer()
result = binerizer.fit_transform(result)

print(result)
print(encoder.classes_)