import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
dataset=pd.read_csv("diabetes.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.describe())
print(dataset['Outcome'].value_counts())
X=dataset.drop(columns='Outcome',axis=1)
y=dataset['Outcome']
print(X)
print(y)
scaler=StandardScaler()
scaler.fit(X)
stand_data=scaler.transform(X)
print(stand_data)
X=stand_data
Y=dataset['Outcome']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,stratify=Y,random_state=2)
classifer =svm.SVC(kernel='linear')
classifer.fit(X_train,Y_train)
X_train_prediction=classifer.predict(X_train)
train_data=accuracy_score(X_train_prediction,Y_train)
print(train_data)
pickle.dump(classifer, open("model1.pkl", "wb"))