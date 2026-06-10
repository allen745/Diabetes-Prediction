# importing Dependencies

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data collection and analysis
# loading the diabetes data set to pandas dataframe
diabetes_dataset = pd.read_csv('C:\\Users\\allen\\OneDrive\\Desktop\\understanding\\data\\diabetes.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# printing first 5 rows of the dataset
print(diabetes_dataset.head())

# printing the shape of dataset
print(diabetes_dataset.shape)                     # (768, 9)

# checking is there any null values
print(diabetes_dataset.isnull().sum())

# statical measures of data frame
print(diabetes_dataset.describe())

# count number of outcomes
print(diabetes_dataset['Outcome'].value_counts())  # 0-->500(non) 1-->268

# mean of each label
print(diabetes_dataset.groupby('Outcome').mean())

# separating label and data
x = diabetes_dataset.drop('Outcome', axis = 1)
y = diabetes_dataset['Outcome']
print(x)
print(y)

# data standardization
scaler = StandardScaler()
print(scaler.fit(x))
standardized_data = scaler.transform(x)
print(standardized_data)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(standardized_data, y, test_size = 0.1, stratify=y,random_state = 2)
print(X_train.shape, y_train.shape, X_test.shape)

# train model
classifier = svm.SVC(kernel = 'linear')
# training the support vector machine classifier
classifier.fit(X_train, y_train)

# modal evaluation
# accuracy score
# accuracy scor on training data3
X_train_predict = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, y_train)
print('training accuracy',training_data_accuracy)

# test data
X_test_predict = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_predict, y_test)
print('testing accuracy',testing_data_accuracy)

# prediction system
input_data = (1,80,55,0,0,19.1,0.258,21)
# changing the input in to numpy array
input_data_as_numpyarray = np.asarray(input_data)

# reshape the array as we are predicting for 1 instance
input_data_reshape = input_data_as_numpyarray.reshape(1, -1)

# standarise input data
std_data = scaler.transform(input_data_reshape)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if prediction == 1:
    print("person is diabatic")
else:
    print("person is not diabatic")








