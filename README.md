# seattle
Transit ridership prediction on the Seattle light rail.

I am keeping this repository private, because the data was granted to me through a public records request.

**8/15: Currently in the process of nicely organizing this repo. Please explore different sections of the project by reading the notebooks in the /notebooks folder.**

**Please refer to Models_Aug02.ipynb for one giant notebook with all work done through 8/8/22.**

## Partial story (8/8)

I fit some seasonal trends to the data, and I also made clear visualizations of the dataset. Bonus: PCA on stations has a nice visual and interpretation.

![figW](/images/imageW.png)

Above is EDA showing that we have more observations during weekday rush hours. Sound Transit runs more vehicles at those times.

![figX](/images/imageX.png)

Above we see that the mean passenger values have maxima at weekday rush hours. Mon-Thurs have a different pattern than Sat-Sun. Friday is somewhere in between.

![figY](/images/imageY.png)

Above we see mean passenger values after we remove seasonal and week-period trends. For this plot, that comes down to adjusting the height of each day-of-week's curve, without changing its shape.

I'd like to do something more interesting with the trend-removed data. Coming soon...

How did I fit the trends? See below for reference.

![figA](/images/imageA.png)

![figB](/images/imageB.png)

![figC](/images/imageC.png)

![figD](/images/imageD.png)

![figE](/images/imageE.png)

Bonus! PCA on stations reveals interesting patterns.

![figZ](/images/imageZ.png)

Notice that if you connect the point in the order the train visits them, it's a butterfly shape.

Conclusions from plot:
- PC1 mostly captures city center stations versus outer stations.
- PC2 mostly captures Northbound vs Southbound.

Conclusions from loadings:
  - (from PC2) On weekend mornings, Northbound stops may be more crowded than Southbound.
  - (from PC1) On weekend mornings, city center stops may be more crowded than outer stops.
  - (from PC2) On weekday 4-6pm times, Southbound stops may be more crowded than Northbound stops.
  - (from PC1) On weekday late-nights, outer stops may be more crowded than city center stops.
    - This is explained by the the airport stop and the college stop at either end - both would spur late-night crowds!






## Summary of work so far

- Data cleaning
  - Which datapoints can be trusted? If not, remove them.
    - Removed impossibly crowded observations (>210 passwithin)
    - Removed routes with >20 stops (the operator forgot to reset the counter)
    - Removed the 0.2% of routes that do not follow the standard calculation (lastwithin + in - out = currwithin)
    - Removed routes where the train visited stations out-of-sequence
  - Extracted features for time dimensions and booleans for crowding levels
    - Focus on 'Crowded' feature - 74 threshold  corresponds to nbr of seats in vehicle
  - Created IDs for stations and directions

- EDA
  - Crowded state happens disproportionately on weekdays 7-9a and 4-6p - especially Weds and Thurs.
  - Crowded state happens more in Jul-Sep, and less in Jan-Mar
  - On passwithin, fit sinusoidal curve with year period - min of curve is 31 in January and max of 38 in July.

- PCA
  - Find mean passwithin by Hour, Day of Year, and Station-direction
  - PCA on stations
    - Each station-direction represented via passenger patterns by hour by day
    - Remove 1am, 2zm, 3am, 4am hours due to too few observations (many NaNs)
    - Fill remaining NaNs using interpolation (previous hour and next hour)
    - PCA brings 7000-ish dimensions down to just 2 components representing 83% of the variance
    - Plot stations in PC-space - reveals butterfly shape
    - Add more variables to plot: find Northbound wing and Southbound wing of butterfly.
    - PC1 mostly captures city center stations versus outer stations.
    - PC2 mostly captures Northbound vs Southbound.
    - Further conclusions based on PC loadings are in the latest notebook.
  - PCA on days
    - Each day represented via passenger patterns by hour by station-direction
    - Fill NaNs with 0.
    - PCA brings 750 dimensions down to 2 components representing 40% of the variance
    - Plot days in PC-space - reveals 2 clusters byeye
    - Explore clustering methods to capture this; settle on Gaussian Mixture
    - Add more variables to plot: find Workday cluster and Weekend-Holiday cluster
    - identified non-Weekend non-Holidays in latter claster as either snowstorms or near-holidays such as July 5th or Black Friday.

- Predictive modeling
  - Test/train datasets; cross validation and readout functions with precision, recall, and f1-score
  - Baseline models
    - Classification: predict whether target Crowded is True or False.
    - One-hot features: station, direction, hour of day, day of week, month
    - Logistic Regression and Random Forest
    - Best f1 score from Random Forest with class_weight = 'balanced'. Scores at 0.50.
    - Precision/recall tradeoffs for different models and class_weight parameter.
  - Attempts at creative scoring
    - Wanted to incorporate regression - would allow using sinusoid-subtracted values
      - Fit linear regression and scored as classfication - 0 or near-0 predictions of >74 - not useful
    - Tried removing 70-79 passenger values to help classifier focus on the important distinction - success, but need more proof that this is best

### Next steps
  - Compare predictive models
    - Try using features from PCA and seasonal trend
    - Try regression-based approaches and classification-based approaches
      - Regression-based approaches may need to add a standard error to prediction in order to improve recall
    - Try training on borderline-removed dataset, but compare to other models using apples-to-apples evaluation
    - Try classification with 3 different classes
  - Gather key figures in the same place
  - Make more visual figures to show my results
  - Wish list: PCA on Crowded variable; Day PCA using interpolation; spectral analysis of periodic trends
