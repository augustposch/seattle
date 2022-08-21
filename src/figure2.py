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



variables = ['passwithin','Removed01','Removed02A']
agg_df = df.groupby(['DOW','Hour'])[variables].agg(['mean','median','count'])

agg_df2 = df.groupby(['DOW','Hour'])['WOY'].nunique() # How many different weeks did this combo have any trains running?
agg_df = pd.concat([agg_df,agg_df2], axis=1).reset_index()

filt = agg_df['WOY'] > 13 # Only DOW-hour combos that occured more than 13 weeks of the year



sns.set_theme(style='whitegrid')
ax = sns.lineplot(data=agg_df[filt], x='Hour', y=('passwithin','mean'), hue='DOW',
                palette='gist_rainbow')
ax.set_title('Mean passengers by hour-of-day and day-of-week',size=14)
ax.set_ylabel('Mean passengers')

ax.legend_.set_title('Day of Week')
new_labels = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
texts = ax.legend_.texts
for t, label in zip(texts, new_labels):
    t.set_text(label)
sns.move_legend(ax, "upper left", bbox_to_anchor = (1,1))
plt.savefig('../images/reproduce/MeanPass1.png', bbox_inches='tight')