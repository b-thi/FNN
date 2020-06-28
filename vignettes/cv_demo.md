# Crossvalidation Demo

Crossvalidation (cv) is important to quantify the uncertainty in any models predictive power. The FNN package offers an easy and simple implementation of cv for users to apply in their own contexts. 

First, we'll read in the data and load some libraries.
``` r
# Libraries
library(fda)

# Loading data
data("daily")
```
Before doing anything else, we're going to do some pre-processing to get our functional observations. Remember, we can let the fnn.fit() function do this for us (as see in the classification example) but, for this example, let's do it ourselves.
``` r
# Creating functional data
nbasis = 65
tempbasis65  = create.fourier.basis(c(0,365), nbasis)
timepts = seq(1, 365, 1)
temp_fd = Data2fd(timepts, daily$tempav, tempbasis65)
```
In the chunk of code above, we are creating functional observations using a 65 term Fourier basis expansion. The Data2fd() function converts the raw data into the functional data objects that we need. In particular, we are concerned with the coefficients defining each of the 35 functional observations here (one set for each of the 35 cities). Let's not extract out those functional observations and get them into the format required for the fnn.fit() function:
``` r
# Non functional covariate
weather_scalar = data.frame(total_prec = apply(daily$precav, 2, sum))

# Setting up data to pass in to function
weather_data_full <- array(dim = c(nbasis, 35, 1))
weather_data_full[,,1] = temp_fd$coefs
scalar_full = data.frame(weather_scalar[,1])
total_prec = apply(daily$precav, 2, mean)
```
Above, we first define a scalar covariate in the appropriate matrix format. The scalar covariate is the total amount of rain over the year in each of the cities (feel free to leave this covariate out). We then store the functional curves (and in particular, the coefficients) from the functional object, into a tensor. Here, the tensor is actually just a matrix because we have only 1 functional covariate (temperature curves for the year). Finally, let's run our cross-valiation function!
``` r
# cross-validating
cv_example <- fnn.cv(nfolds = 5,
                     resp = total_prec,
                     func_cov = weather_data_full,
                     scalar_cov = scalar_full,
                     domain_range = list(c(1, 365)),
                     learn_rate = 0.001)
```
We can adjust the number of folds with nfolds option. We leave the model at its defaults but you can definitely improve the model fitting across the folds with a more appropriate number of layers, neurons, etc.
