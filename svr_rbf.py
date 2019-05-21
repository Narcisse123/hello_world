import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR


dataset = pd.read_excel('Foundries data set.xls')
print('dataset read successfully')
new_column = np.zeros((dataset.shape[0], 1), dtype=np.int)

for i in range(len(new_column)):
    new_column[i, 0] = i+1

new_dataset = np.array(dataset)
new_dataset = np.concatenate((new_column, new_dataset), axis=1)

np.random.shuffle(new_dataset)

X = new_dataset[:, 0]
Y = new_dataset[:, 3]

training_size = int(0.7*len(X))
training_set = X[:training_size].reshape(training_size, 1)
training_labels = Y[:training_size]

test_set = X[training_size:]
test_labels = Y[training_size:]
test_set = test_set.reshape(len(test_set), 1)

clf = SVR(kernel='rbf', gamma='scale', C=10, epsilon=0.5)
clf.fit(training_set.reshape(len(training_set), 1), training_labels)

training_score = clf.score(training_set, training_labels)
test_score = clf.score(test_set, test_labels)
print('Training Score : ', training_score)
print('Test Score : ', test_score)

training_prediction = clf.predict(training_set)
test_prediction = clf.predict(test_set)

plt.subplot(2, 1, 1)
plt.plot(training_set, training_labels, 'bo')
plt.plot(training_set, training_prediction, 'ro')
plt.title('RBF Kernel Regression : Training Data')
plt.legend(['Data', 'Predicted'])

plt.subplot(2, 1, 2)
plt.plot(test_set, test_labels, 'bo')
plt.plot(test_set, test_prediction, 'ro')
plt.title('RBF Kernel Regression : Test, Data')
plt.legend(['Data', 'Predicted'])

plt.show()



