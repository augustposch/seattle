#!/usr/bin/env python
# coding: utf-8

# # Cycles
# 
# This notebook explains how I fit different cycles to the passenger data, and provides nice visuals.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy import optimize


# In[2]:


import sys
sys.path.append('../src/')
import aposch_functions as aposch


# ## Get df04_train dataset

# In[3]:


# read in data
data2019 = pd.read_csv('../data/2019_RAW_APC_Data.csv.gz')

# get df04
df04 = aposch.clean_data(data2019)

# split into test/train sets
df04_train, df04_test = train_test_split(df04,
                                         test_size=0.2,
                                         random_state=19)


# ## Fit seasonal cycle (sinusoidal curve)
# 

# Let's get started. Find the mean crowdedness by day (agnostic of station or of hour):

# In[4]:


pw_agg_doy = df04_train.groupby(['DOY'])['passwithin'].mean()
cr_agg_doy = df04_train.groupby(['DOY'])['Crowded'].mean()
pw_agg_woy = df04_train.groupby(['WOY'])['passwithin'].mean()
cr_agg_woy = df04_train.groupby(['WOY'])['Crowded'].mean()


# Fit a sine curve with period 1 year:

# In[5]:


def sinu(x, a, c, d):
    b = 6.28/52 # forces the period to be 1 year
    return a * np.sin(b * (x - c)) + d

params, params_covariance = optimize.curve_fit(sinu, pw_agg_woy.index, pw_agg_woy,
                                               p0=[3, 13, 33])


# Plot it by week:

# In[6]:


x_data = pw_agg_woy.index
y_data = pw_agg_woy
y_sinu_year = pd.Series(data=sinu(x_data, params[0], params[1], params[2]),
                   index=x_data)

#data manipulation:

# In[7]:


# create a day-level dataframe with the columns we need
byday = aposch.create_byday(pw_agg_doy, pw_agg_woy, y_sinu_year)


# Now a plot at the day of year level:

# In[20]:


x_data = byday['Date']
y_data = byday['DayMeanPassw']
y_sinu_year = byday['Sinu01Passw']



# In[19]:


x_data = byday['Date']
y_data = byday['Removed01']

plt.figure(figsize=(16,4))
plt.plot(x_data, y_data)
plt.xlabel('Date')
plt.ylabel('Mean passengers minus seasonal trend')
plt.title('After removing the year-period seasonal cycle, a week-period cycle remains.', size=14)
plt.savefig('../images/reproduce/DateSeasonal2.png', bbox_inches='tight')

