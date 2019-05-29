
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR


dataset = pd.read_excel('Foundries data set.xls')
print('dataset read successfully')
#new_column = np.zeros((dataset.shape[0], 1), dtype=np.int)

#for i in range(len(new_column)):
    #new_column[i, 0] = i+1

#new_dataset = np.array(dataset)
#new_dataset = np.concatenate((new_column, new_dataset), axis=1)

#np.random.shuffle(new_dataset)

dataset.drop("INDEX", axis=1, inplace=True)
dataset.drop("REF_DATE", axis=1, inplace=True)
dataset.drop([0],axis=0,inplace=True)


x=dataset['Unnamed: 2']

new_array=np.zeros((135,5))

for i in range(len(x)):
    for j in range(0,5):
        new_array[i][j]=x[i+1]

for i in range(0,134):
    new_array[i][1]=new_array[i+1][1]

for i in range(0,133):
    new_array[i][2]=new_array[i+2][2]

for i in range(0,132):
    new_array[i][3]=new_array[i+3][3]


new_array = pd.DataFrame(new_array, columns=['x1', 'x2', 'x3', 'x4', 'change'])


new_array.drop([131,132,133,134],axis=0, inplace=True)

print((new_array))

X = new_array
Y = new_array['change']

print(new_array.shape)
print(new_array)


training_size = int(0.77*len(X))
training_set = X[:training_size]
training_labels = Y[:training_size]

test_set = X[training_size:]
test_labels = Y[training_size:]
test_set = test_set
clf = SVR(kernel='rbf',gamma=2.1510, C=16.970, epsilon=0.1, max_iter=1000)
clf.fit(training_set, training_labels)

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

