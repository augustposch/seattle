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


filt = (df['stadir_ID']==1211) & (df['stop arrival time'].between('2019-08-24 04:00','2019-08-25 04:00'))

sns.set(rc={'figure.figsize':(8,4)},style='whitegrid')
ax = sns.scatterplot(data=df[filt], x='stop arrival time', y='passwithin',
                      markers=('o'),
                      color=(0,0.4,0.2,0.7),
                      s=64)
ax.axhline(74, linestyle='--', color='gray', label='Seats in Vehicle') # horizontal line at Crowded threshold
ax.set_title('Observations at Pioneer Square Southbound on Saturday, 24 Aug 2019', size=14)
ax.set_ylabel('Passengers in Vehicle')
plt.ylim(bottom=0)
ax.set_xlabel('Time')
start, end = ax.get_xlim()
plt.xlim((end-0.90871, end))
xticks = ax.get_xticks()
xlab = ['4 am','6 am','8 am','10 am','Noon','2 pm','4 pm','6 pm','8 pm','10 pm','Midnight']
plt.xticks(ticks=xticks, labels=xlab, rotation=90)

ax.legend()
sns.move_legend(ax, "upper left", bbox_to_anchor = (1,1))

plt.savefig('../images/reproduce/ScatterPioneerSat.png', bbox_inches='tight')