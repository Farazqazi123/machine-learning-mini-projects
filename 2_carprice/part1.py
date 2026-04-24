import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

df=pd.read_csv("car data.csv")

print(df.isnull().sum())  #give that data contain null or not

#print(df.Fuel_Type.value_counts())
#print(df.Seller_Type.value_counts())
#print(df.Transmission.value_counts())


#DOING ENCODING OF CATEGORICAL DATA
df_encoded=pd.get_dummies(df,columns=['Fuel_Type','Seller_Type','Transmission'],drop_first=True)
print(df_encoded.head())

x= df_encoded.drop(['Car_Name','Selling_Price'],axis=1)
y=df['Selling_Price']


x_train,  x_test , y_train , y_test= train_test_split(x,y ,test_size=0.1,random_state=2)

scaler=StandardScaler()
scaled_xtrain=scaler.fit_transform(x_train)
scaled_xtest=scaler.transform(x_test)


model=LinearRegression()

model.fit(scaled_xtrain,y_train)

prediction=model.predict(scaled_xtest)

#R scored value
errorscore=metrics.r2_score(y_test,prediction)
print(errorscore)

plt.scatter(y_test, prediction)
# This adds a red diagonal line for comparison
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2) 
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()