import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import aposch_functions as aposch

# read in data
data2019 = pd.read_csv('../data/2019_RAW_APC_Data.csv.gz')

# get df04
df04 = aposch.clean_data(data2019)

# split into test/train sets
df04_train, df04_test = train_test_split(df04,
                                         test_size=0.2,
                                         random_state=19)

# get cycle-related features
df05_train = aposch.do_cycles_processing(df04_train)

df = df05_train


# ## Visualize some Crowded (>74 passengers) observations
# 
# - choose a particular station
# - choose a particular week
# 
# Wish list:
# - horizontal line at 74
# - nicer legend titles

# In[18]:

# In[41]:


filt = (df['station name']=='Pioneer') & (df['WOY']==34)


# In[77]:


sns.set(rc={'figure.figsize':(12,4)},style='whitegrid')
ax = sns.scatterplot(data=df[filt], x='stop arrival time', y='passwithin',
                      style='stadir_ID', 
                      markers=('o','^'),
                      alpha=0.5,
                      s=64,
                      hue='DOW',
                      palette='gist_rainbow')
# draw horixontal line at Crowded threshold - currently doesn't work
# ax.plot(['2019-08-19', '2019-08-26'], [74, 74], 'k-', lw=2)
ax.set_title('Observations at Pioneer Square station in late August', size=16)
ax.set_ylabel('Passengers in vehicle')
ax.set_xlabel('Date and time')

handles, labels = ax.get_legend_handles_labels()
new_labels = ['- Day of Week -','Monday','Tuesday','Wednesday','Thursday',
              'Friday','Saturday','Sunday','- Direction -','Southbound','Northbound']
ax.legend(handles, new_labels)
sns.move_legend(ax, "upper left", bbox_to_anchor = (1,1))

plt.show()
plt.savefig('../images/ScatterPioneer1.png', bbox_inches='tight')


# Looking at the above plot gives us some insights.
# - there more Crowded (>74 passengers) observations this station-week than in the dataset in general. By eye it appears that about a quarter of all these observations in Crowded.
# - the most-crowded times were weekday mornings and weekday evenings
# - on weekday mornings, all the most crowded trains were northbound
# - on weekday evenings, all the most crowded trains were southbound
# - on Tuesday, Friday, and Saturday, there were also a few very crowded northbound trains around 11pm. (Perhaps there was an event Tuesday night? A quick Google didn't turn up anything.)

# ## Some extra plots I made a couple months ago

# In[ ]:




