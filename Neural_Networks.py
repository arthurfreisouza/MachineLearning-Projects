import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score




data_set = load_digits()

X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size = 0.2, random_state = 33)

NN = MLPClassifier()
NN.fit(X_train, y_train)

y_pred = NN.predict(X_test)


acc = accuracy_score(y_test, y_pred) * 100

acc_cm = confusion_matrix(y_test, y_pred)


print(f'Accuracy : {acc}')
print(f'Confusion matrix : {acc_cm}')