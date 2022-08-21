import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_scores = np.array(pd.read_csv('../data/KNN_CV_train_scores.csv',header=None))
test_scores = np.array(pd.read_csv('../data/KNN_CV_test_scores.csv',header=None))

train_scores_mean = np.mean(0-train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(0-test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

param_range = [5,25,50,75,100,125,150,175,200]

plt.title('Validation Curve for KNN',size=14)
plt.xlabel('Number of neighbors k')
plt.ylabel('Root Mean Squared Error')
lw = 2
plt.plot(
    param_range, train_scores_mean, label="Training error", color="darkorange", lw=lw
)
plt.plot(
    param_range, test_scores_mean, label="Cross-validation error", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)

plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")

plt.savefig('../images/reproduce/ValidationCurveKNN.png', bbox_inches='tight')
