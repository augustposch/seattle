# seattle
Transit ridership prediction on the Seattle light rail.

I am keeping this repository private, because the data was granted to me through a public records request.

**Please explore different sections of the project by reading the individual notebooks in the /notebooks folder.**

## The basics of our dataset

This dataset is... *describe more here*

1. To get familiar with the dataset, let's look at **which days and times we have observations**:

![fig1](/images/CountObs1.png)

We have more observations during weekday rush hours. Sound Transit runs more vehicles at those times.

2. Let's look at the **mean passengers** grouped by day of week and hour of day. This mean does not care about station or direction - all observations in a day-of-week-and-hour-of-day bucket are used to compute that bucket's mean, regardless of which station or direction they were.

![fig2](/images/MeanPass1.png)

We see that the mean passenger values have maxima at weekday rush hours. Mon-Thurs have a different pattern than Sat-Sun. Friday is somewhere in between.

3. Let's figure out what kinds of times we might see **crowded observations**. Here's a week in late August 2019 at the Pioneer Square station. Keep in mind each rail vehicle has 74 seats plus standing room.

![fig3](/images/ScatterPioneer2.png)

Looking at the above plot gives us some insights.
- There are more observations with >74 passengers for this station-week than in the dataset in general. By eye it appears that about a quarter of all these observations are >74 passengers, whereas other EDA revealed that only 10% of all observations are >74 passengers.
- The most-crowded times were weekday mornings and weekday evenings
- On weekday mornings, all the most crowded trains were northbound. I assume these crowds are commuters.
- On weekday evenings, all the most crowded trains were southbound. I assume these crowds are the commuters again.
- On Tuesday, Friday, and Saturday, there were also a few very crowded northbound trains around 11pm. I assume these crowds are people having a fun night out. (Perhaps there was an event Tuesday night? A quick Google didn't turn up anything.)

Below are a couple zoomed-in versions of the plots above. The first is only the Wednesday southbound observations, and the second is only the Saturday southbound observations. We see that the Wednesday southbound observations are more crowded during the evening commute hours.

![fig3B](/images/ScatterPioneerWeds.png)

![fig3C](/images/ScatterPioneerSat.png)

## Understanding our entire dataset's patterns

4. Let's find the **seasonal pattern** in crowdedness. A sinusoidal curve is common practice for a seasonal cycle, so that's what we'll fit.

![fig4](/images/DateSeasonal1.png)

5. Here is **what's left in the data** after we remove that seasonal cycle.

![fig5](/images/DateSeasonal2.png)

Here I'll stop and address what you may be thinking: October 26-27 and October 12-13 were big dips in passengers. These dips in passengers came from planned construction that closed the downtown section of the line. https://www.soundtransit.org/get-to-know-us/news-events/news-releases/construction-will-close-light-rail-downtown-seattle-three

6. Finally, note that we can remove the **week-period pattern** (i.e. remove the day-of-week specific mean for each day of the week) to get this:

![fig6](/images/DateWeekPrd1.png)

It looks mostly like noise to my eyes, which means we did a good job removing the repeating patterns (year-long and week-long, as explained above.)

7. Let's take a different tack, and try to **understand days better**. I used a dataset of mean passengers, grouped by day of year, hour, and station-direction. I had one row for each day of the year, and collapsed the variance of the 768 columns down to just two PCA components.

![fig7](/images/PCA_days.png)

We see that one cluster is commuter days - every day in this cluster is a weekday. The other cluster is non-commuter days - this cluster consists of weekends and similar days, like July 5th, Black Friday, and snowstorms. *add more explanation here*

8. Let's take yet a different tack, and try to **understand stations better**. I used a dataset of mean passengers, grouped by station-direction, day of year and hour. I had one row for each station-direction, and collapsed the variance of the 7000 columns down to just two PCA components.

![fig8](/images/PCA_stations.png)

Notice that if you connect the points in the order the train visits them, it's a butterfly shape.

Conclusions from plot:
- PC1 mostly captures city center stations versus outer stations.
- PC2 mostly captures Northbound vs Southbound.

Conclusions (preliminary) from loadings:
  - (from PC2) On weekend mornings, Northbound stops may be more crowded than Southbound.
  - (from PC1) On weekend mornings, city center stops may be more crowded than outer stops.
  - (from PC2) On weekday 4-6pm times, Southbound stops may be more crowded than Northbound stops.
  - (from PC1) On weekday late-nights, outer stops may be more crowded than city center stops.
    - This is explained by the the airport stop and the college stop at either end - both would spur late-night crowds!

## Predictive modeling

For simplicity, we restrict our dataset to Pioneer Square station going southbound. We'll be predicting the number of passengers in each arriving vehicle, based on previous observations of Pioneer Southbound arrivals over the past 2 1/2 hours.

### Understanding Pioneer Southbound

From section 3 above we saw visuals as to what the Pioneer Southbound observations look like. Here they are again:

![fig3D](/images/ScatterPioneerWeds.png)

![fig3E](/images/ScatterPioneerSat.png)

I chose Pioneer because it's a station that does consistently have crowded vehicles, much more so than other stations. Transit riders will be extra curious to know if their arriving train at Pioneer will be crowded.

Here are the Exploratory Data Analysis plots, but just for Pioneer.

![fig1PS](/images/PSCountObs1.png)

The above plot shows that the count of observations folows that same pattern as the entire dataset shown in Section 1 - more observations during weekday rush hours, because the Sound Transit runs more vehicles at that time.

![fig2PS](/images/PSMeanPass1.png)

The above plot shows that, on average we do expect the number of passengers to be >74 on weekdays from 4pm to 8pm.

### Setting up our dataset for machine learning

Our targets are observations of the number of passengers in each vehicle. For each target that happens at time t, we construct ten features using the observations from (t - 150 minutes) through t. Each feature is a fifteen-minute period, and the value of the feature is the mean passengers from within that time period.

As we try to find the best model, we'll restrict ourselves to the first 8 features, in order to leave a 30-minute buffer between the time of the features and the time of the prediction. This way, transit riders could use the model in an app. 

I used Persistence as our baseline model. This means we simply use the value of the last feature (i.e. mean crowdedness over the last fifteen minutes) as our prediction. We'll do one Persistence model where it really is that last feature, and a second Persistence model where the one feature is the third-to-last feature, i.e. the fifteen-minute period ending 30 minutes ago.

I tried a Linear Regression, Random Forest, Extra Trees and K-Nearest Neighbors models on both the eight-feature dataset (t-150 through t-30) and a four-feature dataset (t-90 through t-30). I used 5-fold cross-validation to get score estimates and standard errors. I used a few different scores: RMSE helps describe the model's performance in pur regression terms; precision, recall, f1-score, and confusion matrix describe the model's performance in classification terms, where we threshold our y values at 74 to understand when the vehicle's seats are full.

### Results

The best model, in terms of all cross-validation scores, was a k-nearest-neighbors regressor with k=225. It performed significantly better than either Persistence model. 

Persistence from [t-15,t] period:  
f1 0.69 with standard error 0.01  
rmse 27.38 with standard error 1.93  

Persistence from [t-45,t-30] period:  
f1 0.65 with standard error 0.02  
rmse 29.50 with standard error 1.80  

KNN with k=225 from all eight periods: (Best model)  
f1 0.70 with standard error 0.01  
rmse 24.80 with standard error 1.36  

Let's make some bar charts showing all the results:

![figX1](/images/rmse.png)
![figX2](/images/f1.png)
![figX3](/images/recall.png)
![figX4](/images/precision.png)

Model names:
- pers, linreg, forest, extrees, and knn refer, respectively, to Persistence, Linear Regression, Random Forest, Extra Trees and K-Nearest Neighbors.
- 00m refers to using the feature that ended 0 minutes prior to the target.
- 30m refers to using the feature that ended 30 minutes prior to the target. (pers_30m is a good baseline for the context of a transit rider using an app.)
- 8f refers to the eight-feature dataset - eight fifteen-minute periods from t-150 through t-30.
- 4f refers to the four-feature dataset - four fifteen minute periods from t-90 through t-30.

Conclusions:
- The eight-feature dataset always outperformed the four-feature dataset when model type is held constant. Therefore, a hypothetical app should definitely take advantage of data going back 150 minutes.
- The K-Nearest-Neighbors eight-feature model performed the best overall. It was either the best or near-best in RMSE, f1, and precision. In all metrics, it clearly beat the 30-minute persistence model, and also narrowly beat the 0-minute persistence model.
- The Extra Trees model was the best for recall, on par with the not-real-world-applicable 0-minute persistence model.
- These predictions are decent, but still not great for a transit rider. RMSE of 25 very-roughly means that predictions could be off by 25 people on a regular basis. Precision of 0.7 means that 70% of the times the model predicted Crowded (>74 passengers), it was correct. Recall of 0.7 means that 70% of the times the train was Crowded, the model predicted so.
- As a next step we could use feature engineering (e.g. day of week, month, holiday, etc) to improve the model.
- As another next step, we could try a different station or direction, and see if our findings hold at in that different context.
- We are well on our way to an effective app to help transit riders choose when to ride the train.


#### Appendix

A1. Here's the validation curve for K Nearest Neighbors. Why did I choose k=100? It's because for k>=100, it's clear that RMSE has reached its best-possible value. See plot below: (the transparent fill radius is one standard error)

![figA1](/images/ValidationCurveKNN.png)

In this plot, low values of k are overfitting, but as k increases the overfitting decreases until we have an appropriately-fitted model. We can see how k=50 is clearly better than k=25, but once you get beyond k=75 you have diminishing returns. Furthermore, at k=100, the CV RMSE has gotten basically as close as it can to the training RMSE; therefore, we can't expect much more improvement in CV RMSE.

A2. Here's a histogram of the times of the dataset we used for predictive modeling. It also shows which observations had to be culled due to too many NaN features - mostly early-morning ones were removed.

![figA2](/images/PM_kept_thrown_obs.png)

----

