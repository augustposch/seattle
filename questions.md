
# Questions

My comments/questions on August's repo: https://github.com/augustposch/seattle

### Comments/questions/recommendations -- 16 Aug

* Figure 1 -- # of observations
  * number of observations == sum over every station, all trains going in each direction
  * Title should be something like: Number of observations -- sorted by hour of day and day of week
    * Vertical axis shouldn't be density -- should be something computed -- too hard to know what it is
    * the smoothness of the KDEs is artificial and confusing
  * Make the result in Figure 1 easily reproducible
    * Giant notebook is not a good thing for a repo
* Figure 2 -- Passengers count
  * Plot of # of passengers in a vehicle
    * Each "observation" in Figure 1 includes a count of the number of passengers
    * This plot shows the average number of passengers averaged over all stations and all trains going in each direction
  * This mean is averaged over all stations, but grouped by day of week and hour of day
  * Maybe this should be the "mean" -- however you define it, the "mean" should be unambiguous
* Figure 3 -- Seasonal model
  * It's not a trend -- it's a model or a cycle
* Figure 4 -- Residual
  * Show the "residual" after removing this "seasonal model"
* Figure 5 -- Crowded
  * Need one or more examples of the thing you're trying to predict
  * It might make sense for this to be Figure 3
  * Where are observations of "crowded"? -- This could be distinguished by color or threshold value
  * Do you need to show the time history of a specific vehicle? -- A: NO!
  * Or do you need to show the time history at a station? -- A: YES!
  * Plot a time series of whatever you want to predict
    * When is it "crowded", what's a typical example?
  * This will give you an idea of variability in thing you're going to try to predict
* Figure 6 -- PCA
  * I don't understand the conclusions you're drawing from the PCA
  * Also not clear on how exactly the calculation was done
* Figure 7 -- Modeling ...???
