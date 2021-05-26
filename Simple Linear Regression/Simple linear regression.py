#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


data=pd.read_csv('marksdata.csv')


# In[4]:


data


# In[26]:


X=data['Hours']


# In[27]:


type(X)


# In[28]:


#converting pandas into numpy array
X=X.values


# In[29]:


type(X)


# In[30]:


X.shape


# In[38]:


#reshaping into 2D
X=X.reshape(9,1)


# In[39]:


X.shape


# In[40]:


y=data['Marks Secured']


# In[41]:


#Linear regression
from sklearn.linear_model import LinearRegression


# In[42]:


#Empty mind
mind=LinearRegression()


# In[43]:


mind


# In[44]:


#training our model to get coefficient/weight
mind=mind.fit(X,y)


# In[45]:


#predicting with unknown data
mind.predict([[12]])


# In[46]:


#to know weight/coeficient after model trained
mind.coef_


# In[47]:


import joblib


# In[49]:


#saving the model
mind=joblib.dump(mind,'marks.pk')


# In[ ]:




