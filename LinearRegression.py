import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('datasets/SupervisedLearning/linear_regression/youtube_channels.csv')

X_train, X_test, y_train, y_test = train_test_split(np.asarray(data.Num_Videos),
                                                    data.Total_Views, test_size=0.20)

X2_train, X2_test, y2_train, y2_test = train_test_split(np.asarray(data.Days_After_Creation),
                                                        data.Total_Views, test_size=0.20)

X3_train, X3_test, y3_train, y3_test = train_test_split(np.asarray(data.Num_Subscribers),
                                                        data.Total_Views, test_size=0.20)

three_d_X_train = np.stack((X_train, X2_train, X3_train), -1)
three_d_X_test = np.stack((X_test, X2_test, X3_test), -1)

model = LinearRegression()
# model.fit(X_train.reshape(-1, 1), y_train)
model.fit(three_d_X_train, y_train)

# prediction = model.predict(X_test.reshape(-1, 1))
prediction = model.predict(three_d_X_test)

print("Prediction: ", prediction)
print("Coefficient (m): ", model.coef_)
print("Intercept (c): ", model.intercept_)

# plt.scatter(X_test, prediction, color='red')
# plt.xlabel("Number Of Videos, Subscribers and Days")
# plt.ylabel("Number Of Views")
# m, c = np.polyfit(X_train, y_train, 1)
# plt.plot(X_train, m * X_train + c)
# plt.show()
