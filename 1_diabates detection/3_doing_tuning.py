import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf','linear']
}


dataset=pd.read_csv("diabetes.csv")
#print(dataset.head(5))

cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] #removing the 0 value for accuracy
for col in cols:
    dataset[col] = dataset[col].replace(0, dataset[col].median())
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

grid = GridSearchCV(svm.SVC(), param_grid, cv=5)
grid.fit(x_train, y_train)

print("Best Params:", grid.best_params_)

model = grid.best_estimator_
pred = model.predict(x_test)

accuracy=accuracy_score(pred,y_test)
print(accuracy)


input_data = (9,165,88,0,0,30.4,0.302,49)

input_df = pd.DataFrame([input_data], columns=x.columns)


standard = scaler.transform(input_df)

inputprediction = model.predict(standard)
print(inputprediction)

if inputprediction[0] ==0:
    print("no the patient is not having diabates")
else:
    print("yes having diabaties")
