# FNN

This package allows the user to build models of the form: f(z, g(x)) where f() is a neural network, z is a vector of scalar covariates, and g(x) are the set of functional covariates. The package is built on top of the Keras/Tensorflow architecture.

For more information on the methodology: https://arxiv.org/abs/2006.09590

## Installation
You can install `FNN` from GitHub with the following commands:

``` r
library(devtools)
install_github("b-thi/FNN")
```

## Example Run

The package functions can be as simple (or complicated) as you want them to be! To illustrate, let's consider the following example:

First, we'll read in the data and load some libraries
``` r
# Library
library(FNN)

# Loading data
tecator = FNN::tecator
```

