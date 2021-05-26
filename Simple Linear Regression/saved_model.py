#!/usr/bin/env python
# coding: utf-8

# In[4]:


import joblib


# In[5]:


mind=joblib.load('marks.pk')


# In[6]:


mind.predict([[3]])


# In[ ]:




