# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The problem statement for developing a neural network regression model involves predicting a continuous value output based on a set of input features. In regression tasks, the goal is to learn a mapping from input variables to a continuous target variable.

## Neural Network Model

![out1](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/9b814fb1-f62c-48b2-b98a-b0c5ae0f755a)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Ragavendran A
### Register Number: 212222230114
```
#DEPENDENCIES:

from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

#DATA FROM SHEETS:

worksheet = gc.open("DL ex 1").sheet1
rows=worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'int'})
df = df.astype({'Output':'int'})
print(df)

df.head()

#DATA VISUALIZATION:

 x = df[["Input"]] .values
 y = df[["Output"]].values

#DATA SPLIT AND PREPROCESSING:

scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

print(x_train)
print(x_test)

#REGRESSIVE MODEL:

 model = Seq([
 Den(4,activation = 'relu',input_shape=[1]),
 Den(6),
 Den(3,activation = 'relu'),
 Den(1),
 ])

 model.compile(optimizer = 'rmsprop',loss = 'mse')
 model.fit(x_train,y_train,epochs=20)
 model.fit(x_train,y_train,epochs=20)

#LOSS CALCULATION:

loss_plot = pd.DataFrame(model.history.history)
loss_plot.plot()

 err = rmse()
 preds = model.predict(x_test)
 err(y_test,preds)

 x_n1 = [[30]]
 x_n_n = scaler.transform(x_n1)
 model.predict(x_n_n)

#PREDICTION:

y_pred=model.predict(x_test)
y_pred

```
## Dataset Information

![out2](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/1e903112-7fa9-414b-ab67-12473e653046)


## OUTPUT

### Head():
![out3](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/9c384a94-b507-4710-b21b-b817cf499950)

### value of X_train and X_test:

![out4](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/7ef11fe0-45c2-4fd1-9f72-32c1bcabf647)

### ARCHITECTURE AND TRAINING:

![out5](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/e22acb7b-f16d-42ef-aa6b-fc5ac0116b20)

![out6](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/98cc7091-06bb-4ea2-a994-c946ca81b14e)


### Training Loss Vs Iteration Plot

![out7](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/fa8c36ae-5a48-48eb-8232-bb1efe4fd92e)

### Test Data Root Mean Squared Error

![out8](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/f459d60c-f3a1-4803-82d8-0112946cc298)

![out9](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/eb2c8cf0-720d-45c1-be58-c054876a26df)

### New Sample Data Prediction

![out10](https://github.com/ragavanayyadurai/basic-nn-model/assets/118749557/cba309b4-ed29-4827-b90b-ed75efb97da6)

## RESULT

A neural network regression model for the given dataset is developed.

