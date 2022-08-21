
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# In[2]:


import sys
sys.path.append('../src/')
import aposch_functions as aposch


# #### Get df04_train dataset

# In[3]:


# read in data
data2019 = pd.read_csv('../data/2019_RAW_APC_Data.csv.gz')

# get df04
df04 = aposch.clean_data(data2019)

# split into test/train sets
df04_train, df04_test = train_test_split(df04,
                                         test_size=0.2,
                                         random_state=19)


# ## Create dataset for PCA

# In[5]:


# Group passwithin by Hour, DOY, and station (stadir_ID). Take the mean.
pw_agg = df04_train.groupby(['Hour','DOY','stadir_ID'])['passwithin'].mean()


# In[6]:


station_view = pw_agg.reset_index().pivot(index='stadir_ID', columns=['DOY','Hour']).sort_index(axis=1, level='DOY')


# In[7]:


station_view.head(3)


# In[8]:


station_view.isna().sum().sum()


# In[9]:


station_view.shape[0] * station_view.shape[1]


# About 8% of entries in this matrix are NaN.

# In[10]:


# Try filling NaNs with 0
station_view_fill0 = station_view.fillna(0)
station_view_fill0.head(3)


# #### Figure out which hours have very few observations, and remove them.
# 
# Below we see that there are very few DOW-stadir combos that have any observations in the 1:00, 2:00 or 3:00 hours. It's also rather scarce in the 4:00 hour.
# 
# Because of this, I will remove the 1am thru 4am hours. This way, the station_view dataframe will have much fewer NaNs.

# In[11]:


dfo = pd.DataFrame({'Hour':[x[0] for x in pw_agg.index],
                    'DOW':[x[1] for x in pw_agg.index],
                    'stadir_ID':[x[2] for x in pw_agg.index]})
dfo.groupby('Hour').count()


# Remove 1 o'clock hour thru 4 o'clock hour:

# In[12]:


hour = pd.Series(data=[x[2] for x in station_view.columns],
                 index=station_view.columns)
filt = ~hour.between(1,4)

# Station view
station_view_filt = station_view.loc[:,filt]


# Fill NaNs by interpolating:

# In[13]:


# Fill NaNs by interpolating before and after
station_view_ipl = station_view_filt.interpolate(method='linear', axis=1)
station_view_ipl.head(3)


# ## Run the PCA algorithm
# 
# Here I'm doing PCA on the interpolation-filled dataframe.

# In[14]:


pca01 = PCA(n_components=0.95, svd_solver='full')

station_pc = pca01.fit_transform(station_view_ipl)



station_pci = pd.DataFrame(data=station_pc[:,[0,1]], index=station_view.index, columns=['PC1','PC2'])

df = station_pci
northbound = [102,203,304,405,506,607,708,809,910,1011,1112,1213,1314,1415,1516,1616]

df['Northbound'] = np.where(df.index.isin(northbound), 1, 0)

df['ID'] = df.index

df['RouteProgress']= np.where(df['Northbound'],
                             np.round(df['ID'],-2)//100,
                             np.round(1700-df['ID'],-2)//100)

df['Station'] = np.round(df['ID'],-2)//100

df


# In[25]:


sns.set_theme()
ax = sns.scatterplot(data=station_pci, x='PC1', y='PC2', hue='Station',
                     style='Northbound', 
                     markers=('o','^'),
                     alpha=1,
                     palette='tab20b',
                     s=64)
handles, labels = ax.get_legend_handles_labels()
legend_order = [0,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,17,18,19]
handles_lo = [handles[x] for x in legend_order]
labels_lo = [labels[x] for x in legend_order]
labels_lo[-3:-1] = ['Direction','Southbound','Northbound']
ax.legend(handles_lo, labels_lo)
sns.move_legend(ax, "upper left", bbox_to_anchor = (1,1))
plt.title('Stations in PC-space', size=14)
plt.savefig('../images/reproduce/PCA_stations.png', bbox_inches='tight')

