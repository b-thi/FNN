# Regression Demo

Here, we present an example of a regression problem. We are concerned with the fat contents of meat samples as a scalar response. We will use the absorbance curves and their derivatives (functional covariates) along with the water contents (scalar covariate) as our predictors.

First, we'll read in the data and load some libraries.
``` r
# loading data
tecator = FNN::tecator

# libraries
library(fda)
```
Before doing anything else, we're going to do some pre-processing to get our functional observations. Remember, we can let the fnn.fit() function do this for us (as see in the classification example) but, for this example, let's do it ourselves.
``` r
# define the time points on which the functional predictor is observed.
timepts = tecator$absorp.fdata$argvals

# define the fourier basis
nbasis = 29
spline_basis = create.fourier.basis(tecator$absorp.fdata$rangeval, nbasis)

# convert the functional predictor into a fda object and getting deriv
tecator_fd =  Data2fd(timepts, t(tecator$absorp.fdata$data), spline_basis)
tecator_deriv = deriv.fd(tecator_fd)
tecator_deriv2 = deriv.fd(tecator_deriv)
```
In the chunk of code above, we are creating functional observations using a 29 term Fourier basis expansion. These functions are available in the fda package. The Data2fd() function converts the raw data into the functional data objects that we need. We are going to use multiple functional covariates as alluded to earlier by using the derivatives of the absorbance curves; we can easily acquire these derivatives by using the deriv.fd function.

Let's now get our scalar covariate
``` r
# Non functional covariate
tecator_scalar = data.frame(water = tecator$y$Water)
```
And our response
``` r
# Response
tecator_resp = tecator$y$Fat
```
We now need to create a tensor containing the functional covariates (as defined by their coefficients) so that it can be passed into the main model function:
``` r
# Getting data into right format
tecator_data = array(dim = c(nbasis, length(tecator_resp), 3))
tecator_data[,,1] = tecator_fd$coefs
tecator_data[,,2] = tecator_deriv$coefs
tecator_data[,,3] = tecator_deriv2$coefs
```
And the last step before building our model, we create a test train split. This will be quite a few lines of code but all of them are just splitting each of the functional and scalar covariates, as well as the response. In this case, we will use the first 165 curves as the training set and use the final 50 as the test set.
``` r
# Splitting into test and train for third FNN
ind = 1:165
tec_data_train <- array(dim = c(nbasis, length(ind), 3))
tec_data_test <- array(dim = c(nbasis, nrow(tecator$absorp.fdata$data) - length(ind), 3))
tec_data_train = tecator_data[, ind, ]
tec_data_test = tecator_data[, -ind, ]
tecResp_train = tecator_resp[ind]
tecResp_test = tecator_resp[-ind]
scalar_train = data.frame(tecator_scalar[ind,1])
scalar_test = data.frame(tecator_scalar[-ind,1])
```
We now build the model:
``` r
# Setting up network
tecator_fnn = fnn.fit(resp = tecResp_train,
                      func_cov = tec_data_train,
                      scalar_cov = scalar_train,
                      basis_choice = c("fourier", "fourier", "fourier"),
                      num_basis = c(5, 5, 7),
                      hidden_layers = 4,
                      neurons_per_layer = c(64, 64, 64, 64),
                      activations_in_layers = c("relu", "relu", "relu", "linear"),
                      domain_range = list(c(850, 1050), c(850, 1050), c(850, 1050)),
                      epochs = 300,
                      learn_rate = 0.002)
```
In this example, we build a 4 layer network each of which contains 64 neurons. We use 300 training iterations and define the 3 functional weights (for the 3 functional covariates) using 5, 5, and 7 basis functions, respectively.

Let's now get some predictions!
``` r
# Predicting
pred_tec = fnn.predict(tecator_fnn,
                       tec_data_test,
                       scalar_cov = scalar_test,
                       basis_choice = c("fourier", "fourier", "fourier"),
                       num_basis = c(5, 5, 7),
                       domain_range = list(c(850, 1050), c(850, 1050), c(850, 1050)))
```
Using this output, we can compare the predictions to the actual results in whatever way we would like! 

And that is pretty much it for this example.
