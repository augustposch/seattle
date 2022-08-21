import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for statistic in ['precision','recall','f1','rmse']:

    df = pd.read_csv('../data/'+statistic+'.csv')

    sns.set_theme(style='whitegrid')
    sns.barplot(data=df,x='index',y='mean',
                linewidth=0,
                color=(0.8,0.85,1))
    plt.errorbar(x=df['index'],y=df['mean'], linewidth=3,
                yerr=df['std'], fmt='none', color=(0,0,0))
    plt.xticks(rotation=45)
    plt.title(statistic.capitalize()+' from 5-fold cross-validation. (Errorbar radius is one standard error.)')
    plt.xlabel('Model name')
    plt.ylabel(statistic.capitalize()+' (mean of 5 folds)')

    plt.savefig('../images/reproduce/'+statistic+'.png', bbox_inches='tight')
    plt.close()