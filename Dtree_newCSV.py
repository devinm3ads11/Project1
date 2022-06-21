#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#loading data file
df = pd.read_csv('newCSV.csv')


# In[2]:


#displays the dataset being used
print(df.head(10))


# In[3]:


#shows the length of the dataset as how many rows and columns there are
df.shape


# In[4]:


target = df['label']
df1 = df.copy()
df1 = df1.drop('label', axis =1)


# In[5]:


X = df1


# In[6]:


print(target)


# In[7]:


md = LabelEncoder()
target = md.fit_transform(target)
target


# In[8]:


y = target


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 50) 
print("Train split input- ", X_train.shape)
print("Test split input- ", X_test.shape)


# In[10]:


from sklearn import tree 
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
print('Decision Tree Classifier Created')


# In[11]:


y_pred = model.predict(X_test)
df1 = pd.DataFrame({'Labels' :y_test, 'Predictions': y_pred})
print(df1.tail(7))


# In[12]:


print ("Algorithm Accuracy is ", accuracy_score(y_test,y_pred)*100)


# In[ ]:




