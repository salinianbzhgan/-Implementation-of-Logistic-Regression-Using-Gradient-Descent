# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the dataset.

2.Define X and Y array.

3.Define a function for costFunction,cost and gradient.

4.Define a function to plot the decision boundary.

5.Define a function to predict the Regression value 

## Program:
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SALINI A
RegisterNumber: 212223220091
*/

```
import pandas as pd
import numpy as np
```
```
dataset=pd.read_csv('Placement_Data.csv')
dataset
```
```
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
```

```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')    
dataset["status"]=dataset["status"].astype('category') 
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes   
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```

```
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
```
```
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred = np.where(h>= 0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```

```
print(y_pred)
```

```
print(Y)
```

```
xnew= np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

```
xnew= np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

![image](https://github.com/user-attachments/assets/8e5a05bb-7dae-42ca-b024-f875234ce6eb)


![image](https://github.com/user-attachments/assets/2060b3ae-2fb7-4d5a-a28b-51f09469129d)


![image](https://github.com/user-attachments/assets/c12086cf-0322-40d1-b300-d25b8bd80941)


![image](https://github.com/user-attachments/assets/ca41b2cd-7e31-4f89-a3fb-3973ff502660)


![image](https://github.com/user-attachments/assets/aa985ab2-513a-49d5-95e9-f3836bb17233)


![image](https://github.com/user-attachments/assets/8837dad1-fd19-4df3-94c9-38368582aebd)


![image](https://github.com/user-attachments/assets/aec9fb0b-ec7f-475f-b2f6-95f55c4d6a13)


![image](https://github.com/user-attachments/assets/0aaabff6-9d0a-48cd-9122-8e5aac06ef8e)


![image](https://github.com/user-attachments/assets/7e3031ff-f8e5-459b-8513-5ff7c6e61ae8)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
