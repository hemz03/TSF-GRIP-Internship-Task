#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("C:/Users/heman/Desktop/dataset.csv")
data.head()


# In[7]:


data.info() 
data.describe()


# In[12]:


sns.scatterplot(x=data['Hours'], y=data['Scores']);


# In[13]:


sns.regplot(x=data['Hours'], y=data['Scores']);


# In[14]:


X = data[['Hours']]
y = data['Scores']


# In[18]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[19]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()


# In[20]:


regressor.fit(train_X, train_y)


# In[21]:


pred_y = regressor.predict(val_X)


# In[23]:


pd.DataFrame({'Actual': val_y, 'Predicted': pred_y})  ## view actual and predicted on test set side-by-side


# In[24]:


sns.kdeplot(pred_y,label="Predicted", shade=True);

sns.kdeplot(data=val_y, label="Actual", shade=True);


# In[26]:


print('Train accuracy: ', regressor.score(train_X, train_y),'\nTest accuracy : ', regressor.score(val_X, val_y) )


# In[27]:


h = [[9.25]]
s = regressor.predict(h)


# In[28]:


print('A student who studies ', h[0][0] , ' hours is estimated to score ', s[0])


# In[ ]:




