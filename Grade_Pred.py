#!/usr/bin/env python
# coding: utf-8

# In[49]:


import torch
import sklearn
import numpy as np
from sklearn import linear_model


# In[5]:


a = np.loadtxt("/content/drive/MyDrive/Colab_Notebooks/Grade_pred/student-mat.csv",dtype="str", delimiter=";")


# In[6]:


training_size = int(395 * 0.8)


# In[7]:


def variable_picker(Var_array):
  x = np.arange(1,396)
  n_features = len(Var_array)
  data_array = np.zeros(n_features*len(x))
  k = 0
  for i in Var_array:
    for j in x:
      data_array[k] = a[j][i]
      k += 1 

  data_array = data_array.reshape(n_features,395).T
  training_dataset = data_array[:training_size]
  test_dataset = data_array[training_size:]

  return training_dataset,test_dataset,n_features


# In[8]:


def Loader_and_spliter(training_dataset,test_dataset,n_features):
  # Test/train Lodaer
  train_loader = torch.utils.data.DataLoader(training_dataset,batch_size=training_size,shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=True)

  # Train data
  train_iter = iter(train_loader)
  train_data_shuffled = train_iter.next()

  x_train = train_data_shuffled[:,:n_features-1]
  y_train = train_data_shuffled[:,n_features-1:]

  # Test data
  test_iter = iter(test_loader)
  test_data_shuffled = test_iter.next()

  x_test = test_data_shuffled[:,:n_features-1]
  y_test = test_data_shuffled[:,n_features-1:]

  return x_train,y_train,x_test,y_test


# ![Image of dataset features](https://user-images.githubusercontent.com/102351774/179286989-5369c17f-648a-4e19-9b50-6bbd40534651.png) 

# When choosing a list of variable try picking integers and not bools or string since the regression will not work with them.

# In[11]:


Var_array = [2,13,14,28,29,30,31,32]
training_dataset,test_dataset,n_features = variable_picker(Var_array)
x_train,y_train,x_test,y_test = Loader_and_spliter(training_dataset,test_dataset,n_features)


linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)

print(f"Accuracy of prediction was {round(acc,3)*100}%")


# In[13]:


Var_array_2 = [13,14,30,31,32]
training_dataset_2,test_dataset_2, n_features = variable_picker(Var_array_2)
x_train_2,y_train_2,x_test_2,y_test_2 = Loader_and_spliter(training_dataset_2,test_dataset_2,n_features)

linear = linear_model.LinearRegression()
linear.fit(x_train_2,y_train_2)
acc = linear.score(x_test_2,y_test_2)

print(f"Accuracy of prediction was {round(acc,3)*100}%")


# In[46]:


print("Coefficients:")
print(*np.round(linear.coef_[0],5), sep=", ")
print(f"y-intercept: {round(linear.intercept_[0],3)} first varible set\n")


print(f"Coefficients:")
print(*np.round(linear.coef_[0],5), sep=", ")
print(f"y-intercept: {round(linear.intercept_[0],3)} second varible set")

