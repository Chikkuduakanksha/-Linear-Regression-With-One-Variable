#!/usr/bin/env python
# coding: utf-8

# #                          Linear Regression With One Variable

# # Predicting home price in new jersey (USA)

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 


# In[88]:


df = pd.read_csv("homeprices.csv")
df


# In[77]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sqr ft)")
plt.ylabel('price(US $)')
plt.scatter(df.area, df.price, color='Blue', marker ='*')
plt.plot(df.area,df.price, color = 'brown' )


# # Build a Model

# In[34]:


# Model name: reg
reg = linear_model.LinearRegression()


# # Fit the Model

# In[36]:


reg.fit(df[['area']], df.price)


# # Predict price of a home with area = 6000 sqr ft

# In[78]:


reg.predict([[6000]])


# In[40]:


reg.coef_


# In[41]:


reg.intercept_


# In[80]:


import warnings
warnings.filterwarnings("ignore")


# # Generate CSV file with list of home price predictions

# In[87]:


d = pd.read_csv('homepricepredict.csv')


# In[84]:


p = reg.predict(d)
p


# In[60]:


d['prices'] = p


# In[68]:


d.to_csv("Prediction.csv", index =False)


# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sqr ft)")
plt.ylabel('price(US $)')
plt.scatter(df.area, df.price, color='Blue', marker ='*')
plt.plot(df.area,reg.predict(df[['area']]), color = 'green' )

