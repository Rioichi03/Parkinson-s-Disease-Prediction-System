import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# loading the heart dataset to a pandas DataFrame
df=pd.read_csv('heart.csv')

# seperating the data and label into X and Y respectively
X=df.iloc[:,0:13]
Y=df['target']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=1)

# Building the model using Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# Save the trained Logistic Regression model with pickle
pickle.dump(classifier, open('heart.pkl', 'wb'))

# Accuracy test
from sklearn.metrics import accuracy_score

# Assuming the classifier is already trained and you have X_test and Y_test
# ...

# Predict on the test set
Y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)

# ... (Previous code for loading data, training, and saving the model)

# Predict on the test set
Y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(Y_test, Y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(Y_test, Y_pred)
print("Recall:", recall)

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ... (Previous code for loading data, training, and saving the model)

# Predict on the test set
Y_pred = classifier.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Visualize the Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.show()




