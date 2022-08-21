import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from pandas.tseries.holiday import USFederalHolidayCalendar


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

# In[4]:


# Group passwithin by Hour, DOY, and station (stadir_ID). Take the mean.
pw_agg = df04_train.groupby(['Hour','DOY','stadir_ID'])['passwithin'].mean()


# In[5]:


day_view = pw_agg.reset_index().pivot(index='DOY', columns=['Hour','stadir_ID'])


# Try filling NaNs with 0
day_view_fill0 = day_view.fillna(0)


# ## Run PCA algorithm

# In[9]:


pca02 = PCA(n_components=0.7, svd_solver='full')

day_pc = pca02.fit_transform(day_view_fill0)



gmm = GaussianMixture(n_components=2, covariance_type='full')

labels = gmm.fit_predict(day_pc[:,[0,1]])


# The Gaussian Mixture looks the best.

# ## Investigate the clusters
# 
# Make a dataframe to investigate this.

# In[17]:


dates2019 = pd.date_range("2019-01-01", periods=365, freq="D")


# In[29]:


# day_pc_investigation

day_pci = pd.DataFrame(data=day_pc[:,[0,1]], index=dates2019, columns=['PC1','PC2'])

cal = USFederalHolidayCalendar()

day_pci['Holiday'] = day_pci.index.isin(cal.holidays())

day_pci['DayName'] = day_pci.index.day_name()

day_pci['DOW'] = day_pci.index.day_of_week

day_pci['GMM_Label'] = labels

day_pci


# Plot in seaborn, coloring by holidays:

# In[19]:


ax = sns.scatterplot(data=day_pci, x='PC1', y='PC2', hue='DOW', style='Holiday',palette='gist_rainbow')
ax.set_title('Days in PC-space', size=14)
sns.move_legend(ax, "upper left", bbox_to_anchor = (1,1))

handles, labels = ax.get_legend_handles_labels()
new_labels = ['Monday','Tuesday','Wednesday','Thursday',
              'Friday','Saturday','Sunday','Non-Holiday','Holiday']
h_order = [1,2,3,4,5,6,7,9,10]
new_handles = [handles[x] for x in h_order]
ax.legend(new_handles, new_labels)
sns.move_legend(ax, "upper left", bbox_to_anchor = (1,1))

plt.savefig('../images/reproduce/PCA_days.png', bbox_inches='tight')

