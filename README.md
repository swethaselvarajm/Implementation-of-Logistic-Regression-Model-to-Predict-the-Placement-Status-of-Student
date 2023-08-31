# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and 
   duplicated() function respectively.
3. LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required 
   modules from sklearn.
7.Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SWETHA.S
RegisterNumber: 212222230155
*/
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1["gender"])
data1['ssc_b']=le.fit_transform(data1["ssc_b"])
data1['hsc_b']=le.fit_transform(data1["hsc_b"])
data1['hsc_s']=le.fit_transform(data1["hsc_s"])
data1['degree_t']=le.fit_transform(data1["degree_t"])
data1['workex']=le.fit_transform(data1["workex"])
data1['specialisation']=le.fit_transform(data1["specialisation"])
data1['status']=le.fit_transform(data1["status"])
print(data1)

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
ORIGINAL DATA:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/0e25bb03-8de2-4a6a-8e69-97593f4bdf58)

DATA AFTER DROPPING UNWANTED COLUMNS:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/9b146f2c-e798-46ee-8dcf-c01134ae7363)

CHECKING THE PRESENCE OF NULL VALUES:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/b61d35fd-2818-4c62-861d-0837503df148)

CHECKING THE PRESENCE OF DUPLICATED VALUES:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/806b49c8-5580-481c-96e2-4bab2c760454)

DATA AFTER ENCODING:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/8aaee3a4-ab89-49b4-8643-7be4a9df186b)

X DATA:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/bc52f14f-8546-4db9-85a4-a05e999cce88)

Y DATA:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/8c8a65df-c0db-46bb-9e0e-824cf799c9bf)

PREDICTED VALUES:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/6836111e-49e2-4900-a6bf-5a7afe93bd21)

ACCURACY SCORE:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/67712acd-f2ac-474e-83f8-010dbef1cb84)

CONFUSION MATRIX:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/4907578a-1f54-4043-abba-6c897f054397)

CLASSIFICATION REPORT:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/8983f574-bd13-4961-a069-124408531603)

PREDICTING OUTPUT FROM REGRESSION MODEL:

![image](https://github.com/swethaselvarajm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119525603/8fd8c9e4-a66f-43ee-a69e-9c86103e8e1c)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
