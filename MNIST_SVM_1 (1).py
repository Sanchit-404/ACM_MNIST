#!/usr/bin/env python
# coding: utf-8

# In[2]:

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import time
import os


# In[12]:


from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', data_home='./')

mnist.keys()

images = mnist.data
targets = mnist.target


# In[13]:


X_data = images/255.0
Y = targets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)


# In[14]:


parameter_C = 5
parameter_gamma = 0.05
classifier = svm.SVC(C=parameter_C,gamma=parameter_gamma)

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(X_train, y_train)
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))


# In[ ]:


expected = y_test
predicted = classifier.predict(X_test)

show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier {}:\n{}\n".format(classifier, metrics.classification_report(expected, predicted)))
     
      
conf_matrix = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n{}".format(conf_matrix))

plot_confusion_matrix(conf_matrix)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


# In[ ]:




