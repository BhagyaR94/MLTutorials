from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, classification_report
import pandas as pd

actual = [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0]
predicted = [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]

# Individually
print(accuracy_score(actual, predicted))
print(f1_score(actual, predicted))
print(precision_score(actual, predicted))
print(confusion_matrix(actual, predicted))

# Combined
report = pd.DataFrame(classification_report(actual, predicted, output_dict=True))
print(report)
