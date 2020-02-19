# make predictions
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, neighbors
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
# Load dataset

dataset = read_csv('dataset.csv', index_col = [0])
#print(dataset.head())


# sns.pairplot(dataset, hue = 'class')
# plt.show()
# Split-out validation dataset
#array = dataset.loc[:,:'Val']

X = dataset[['Hue','Sat','Val']]
data_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
X = data_scaler.fit_transform(X)
y = dataset['Out']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Make predictions on validation dataset
#model = SVC(gamma='auto')
model = LinearRegression()
#model = LogisticRegression()
model.fit(X_train, Y_train)
accuracy = model.score(X_validation, Y_validation)
#Evaluate predictions
print(accuracy*100)
# #print(accuracy_score(Y_validation, predictions))
# #print(confusion_matrix(Y_validation, predictions))
# #print(classification_report(Y_validation, predictions))