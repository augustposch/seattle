import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import sys
sys.path.append('../src/')
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



filt3 = (df['station name']=='Pioneer') & (df['WOY']==34)

sns.set(rc={'figure.figsize':(12,4)},style='whitegrid')
ax = sns.scatterplot(data=df[filt3], x='stop arrival time', y='passwithin',
                      style='stadir_ID', 
                      markers=('o','^'),
                      alpha=0.5,
                      s=64,
                      hue='DOW',
                      palette='gist_rainbow')
ax.axhline(74, linestyle='--', color='gray', label='Seats in Vehicle') # horizontal line at Crowded threshold
ax.set_title('Observations at Pioneer Square station in late August', size=16)
ax.set_ylabel('Passengers in vehicle')
ax.set_xlabel('Date and time')

handles, labels = ax.get_legend_handles_labels()
new_labels = ['Monday','Tuesday','Wednesday','Thursday',
              'Friday','Saturday','Sunday','Number of Seats','Southbound','Northbound']
h_order = [1,2,3,4,5,6,7,11,9,10]
new_handles = [handles[x] for x in h_order]
ax.legend(new_handles, new_labels)
sns.move_legend(ax, "upper left", bbox_to_anchor = (1,1))
plt.savefig('../images/reproduce/ScatterPioneer2.png', bbox_inches='tight')