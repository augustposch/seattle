#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Read and Clean Dataset
# 

# In[386]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from pandas.tseries.holiday import USFederalHolidayCalendar

from scipy import optimize

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# ## Read and clean dataset

# In[3]:


data2019 = pd.read_csv('2019_RAW_APC_Data.csv.gz')


# In[4]:


for col in ['route finish time','route start time','stop arrival time']:
    data2019[col] = pd.to_datetime(data2019[col])


# In[5]:


df = data2019

df['Crowded'] = df['passwithin']>74
df['Supercrowded'] = df['passwithin']>134
df['Capacity'] = df['passwithin']>194
df['UnderNeg5'] = df['passwithin']<-5
df['NegAFew'] = df['passwithin'].between(-5,-1)
df['Over250'] = df['passwithin']>250
df['AllObservations'] = True

df['Crowded000'] = df['passwithin']>0
df['Crowded010'] = df['passwithin']>10
df['Crowded020'] = df['passwithin']>20
df['Crowded030'] = df['passwithin']>30
df['Crowded040'] = df['passwithin']>40
df['Crowded050'] = df['passwithin']>50
df['Crowded060'] = df['passwithin']>60
df['Crowded070'] = df['passwithin']>70
df['Crowded080'] = df['passwithin']>80
df['Crowded090'] = df['passwithin']>90
df['Crowded100'] = df['passwithin']>100
df['Crowded110'] = df['passwithin']>110
df['Crowded120'] = df['passwithin']>120
df['Crowded130'] = df['passwithin']>130
df['Crowded140'] = df['passwithin']>140
df['Crowded150'] = df['passwithin']>150
df['Crowded160'] = df['passwithin']>160
df['Crowded170'] = df['passwithin']>170
df['Crowded180'] = df['passwithin']>180
df['Crowded190'] = df['passwithin']>190
df['Crowded200'] = df['passwithin']>200
df['Crowded210'] = df['passwithin']>210
df['Crowded220'] = df['passwithin']>220
df['Crowded230'] = df['passwithin']>230
df['Crowded240'] = df['passwithin']>240


# In[6]:


df = data2019

df['TOD'] = df['stop arrival time'].dt.time
df['DOW'] = df['stop arrival time'].dt.dayofweek # 0 is Monday, 6 is Sunday
df['DOW_name'] = df['stop arrival time'].dt.day_name()
df['Date'] = df['stop arrival time'].dt.date
df['Hour'] = df['stop arrival time'].dt.hour
df['Minute'] = df['stop arrival time'].dt.minute
df['Minute_od'] = df['Hour'] * 60 + df['Minute']
df['Month'] = df['stop arrival time'].dt.month
df['Month_name'] = df['stop arrival time'].dt.month_name()
df['Season'] = df['stop arrival time'].dt.quarter
df['DOY'] = df['stop arrival time'].dt.dayofyear
df['WOY'] = df['stop arrival time'].dt.isocalendar().week


# In[7]:


df = data2019

# Create a station ID
# Use a dictionary
names = ['Zero','Angle','SeaTac','Tukwila','Rainier','Othello',
         'Columbia','Baker','Beacon','SODO','Stadium','Intl District',
         'Pioneer','University','Westlake','Capitol Hill ','UW ']

for idx, name in enumerate(names):
    df.loc[df['station name']==name, 'sta_ID'] = idx
    df.loc[df['next station']==name, 'nxsta_ID'] = idx
    
df['sta_ID'] = df['sta_ID'].astype('int32')
df['nxsta_ID'] = df['nxsta_ID'].astype('int32')
df['stadir_ID'] = df['sta_ID'] * 100 + df['nxsta_ID']


# In[8]:


# df01 filters out routedones that have >20 stops
rtd = data2019.groupby('routedone').count()['railcar ID']
rtd.name = 'count'
overmuch = rtd[rtd>20]
df01 = data2019[~data2019['routedone'].isin(overmuch.index)]


# In[9]:


# df02 is df01 but without >210 observations
df02 = df01[~df01['Crowded210']]


# In[10]:


# df03 is df02 but without any against-equation trips

against_calc = pd.read_csv('routes_against_calc.csv').iloc[:,0]

df03 = df02[~df02['routedone'].isin(against_calc)]


# In[11]:


# df04 is df03 but without any trips where the train visited stations out of sequence.
# In other words, df04 consists of only normal stadir values.

abnormal = pd.read_csv('abnormal_stadir_routes.csv').iloc[:,0]

df04 = df03[~df03['routedone'].isin(abnormal)]

