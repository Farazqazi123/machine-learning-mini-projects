import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("diabetes.csv")
#print(dataset.head(5))

# num of row and colum
#print(dataset.shape)

#getting stastical measure
#print(dataset.describe())

#number of values
#print(dataset["Outcome"].value_counts())

#--------------------------------------------------------------------
x=dataset.drop(columns='Outcome',axis=1)
y=dataset["Outcome"]


#DATA STANDARDIZATION -->this is for better performance of model

scaler=StandardScaler()
scaled_x=scaler.fit_transform(x)
#print(scaled_x)

#train- test spliting 

x_train,  x_test , y_train , y_test= train_test_split(scaled_x ,y ,test_size=0.2,random_state=2,stratify=y)

model=svm.SVC(kernel='linear')

model.fit(x_train,y_train)
prediction=model.predict(x_test)

accuracy=accuracy_score(prediction,y_test)
print(accuracy)


