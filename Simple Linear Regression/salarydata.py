#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('salarydata.csv')


# In[3]:


data.head()


# In[4]:


X=data['YearsExperience']


# In[5]:


y=data['Salary']


# In[6]:


from sklearn.linear_model import LinearRegression


# In[7]:


mind=LinearRegression()


# In[8]:


mind


# In[9]:


type(X)


# In[10]:


X=X.values


# In[11]:


X=X.reshape(30,1)


# In[12]:


mind=mind.fit(X,y)


# In[13]:


mind.predict([[4]])


# In[14]:


mind.coef_


# In[15]:


mind.intercept_


# In[16]:


#y=c+wx 
#y=target
#c=contsant/bias
#w=weight/coefficent
#=dependent variable


# In[17]:


data.head(12)


# In[18]:


mind.predict([[4.0]])


# In[19]:


#True value=55794
#Predicted value=63592
#y=c+wX
25792.20019866871+9449.96232146*4.0


# In[20]:


import joblib


# In[21]:


joblib.dump(mind,'salarymodel.pk')


# In[ ]:




