import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt





iris_df = load_iris()


df = pd.DataFrame(data = iris_df.data, columns = iris_df.feature_names)
df['target'] = iris_df.target


X_train, X_test, y_train, y_test = train_test_split(df[iris_df.feature_names], df['target'], test_size = 0.25, random_state = 33)


my_classifier = DecisionTreeClassifier(random_state = 33)
my_classifier.fit(X_train, y_train)


y_pred = my_classifier.predict(X_test)

print(f'Accuracy : ', accuracy_score(y_test, y_pred))



fig = plt.figure(figsize=(15, 10))

_ = tree.plot_tree(my_classifier,
                   feature_names = iris_df.feature_names,
                   class_names = iris_df.target_names,
                   filled = True)