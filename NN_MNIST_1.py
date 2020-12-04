#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import os
import csv

os.chdir("E:\\ACM_task\\KNN\\raw_files")


# In[4]:


with open('mnist_train.csv', newline='') as csv_file1:
    
        train_data_lines = csv.reader(csv_file1)
        train_dataset=list(train_data_lines) 
        
        train_matrix=np.array(train_dataset).astype("int")
    
with open('mnist_test.csv', newline='') as csv_file2:
    
        test_data_lines = csv.reader(csv_file2)
        test_dataset=list(test_data_lines)
        
        test_matrix=np.array(test_dataset).astype("int")



# In[5]:



x_train=train_matrix[:,1:]
y_train=train_matrix[:,:1]
x_test=test_matrix[:,1:]
y_test=test_matrix[:,:1]





# In[6]:


X = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))


digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)


m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]

shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]


# In[7]:


def sigmoid(z):
   
    s = 1. / (1. + np.exp(-z))
    return s


# In[8]:


def compute_loss(Y, Y_hat):
  
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L


# In[9]:


def feed_forward(X, params):
   
    cache = {}

    
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]

  
    cache["A1"] = sigmoid(cache["Z1"])

    
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]

    
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache


# In[10]:


def back_propagate(X, Y, params, cache, m_batch):
   
    dZ2 = cache["A2"] - Y

    dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))

    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


# In[11]:


lr=0.5
epochs=30
n_x=784
n_h=64
beta=0.9
batch_size=64

params = {"W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
          "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
          "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
          "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h)}


# In[12]:


# training
batches=1
for i in range(epochs):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):

        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache, m_batch)

        dW1 = (beta * grads["dW1"] + (1. - beta) * grads["dW1"])
        db1 = (beta * grads["db1"] + (1. - beta) * grads["db1"])
        dW2 = (beta * grads["dW2"] + (1. - beta) * grads["dW2"])
        db2 = (beta * grads["db2"] + (1. - beta) * grads["db2"])

        params["W1"] = params["W1"] - lr * dW1
        params["b1"] = params["b1"] - lr * db1
        params["W2"] = params["W2"] - lr * dW2
        params["b2"] = params["b2"] - lr * db2

    cache = feed_forward(X_train, params)
    train_loss = compute_loss(Y_train, cache["A2"])

    cache = feed_forward(X_test, params)
    test_loss = compute_loss(Y_test, cache["A2"])
    print("Epoch {}: training loss = {}, test loss = {}".format(
        i + 1, train_loss, test_loss))
    print(cache)


# In[87]:





# In[ ]:




