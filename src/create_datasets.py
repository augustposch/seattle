# Creates all the datasets and writes them as .csv files in the data folder.
# Don't run this for now, because it creates enough files to put you over the Github repo limit.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import aposch_functions as aposch

# read in raw data
data2019 = pd.read_csv('../data/2019_RAW_APC_Data.csv.gz')

# get df04
df04 = aposch.clean_data(data2019)

# split into test/train sets
df04_train, df04_test = train_test_split(df04,
                                         test_size=0.2,
                                         random_state=19)

# get cycle-related features
df05_train, byday, bydow = aposch.do_cycles_processing(df04_train, return_all_frames=True)

# write all of these to the data folder as .csv files
datasets = [df04, df04_train, df04_test, df05_train, byday, bydow]
names = ['df04', 'df04_train', 'df04_test', 'df05_train', 'byday', 'bydow']
for i in range(len(names)):
    print(names[i])
    path = '../data/' + names[i] + '.csv'
    datasets[i].to_csv(path, compression='gzip')
    
