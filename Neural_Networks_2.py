import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs



n_samples = 200

blob_centers = ([1,1], [3,4], [1, 3.3], [3.5, 1.8])

data, labels = make_blobs(n_samples = n_samples, centers = blob_centers, 
                          cluster_std = 0.5, random_state = 0)

colours = ('green', 'orange', 'blue', 'magenta')
fig, ax = plt.subplots()

for n_class in range(len(blob_centers)):
    ax.scatter(data[labels == n_class][:, 0],
               data[labels == n_class][:, 1],
               c = colours[n_class],
               s = 30,
               label = str(n_class))
    


from sklearn.model_selection import train_test_split

datasets = train_test_split(data,
                            labels,
                            test_size = 0.2)

X_train, X_test, y_train, y_test = datasets



from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver = 'lbfgs',
                    alpha = 1e-5,
                    hidden_layer_sizes = (6,),
                    random_state = 1)

clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
acc


