% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fnn.tune.R
\name{fnn.tune}
\alias{fnn.tune}
\title{Tuning Functional Neural Networks}
\usage{
fnn.tune(
  tune_list,
  resp,
  func_cov,
  scalar_cov = NULL,
  basis_choice,
  domain_range,
  batch_size = 32,
  decay_rate = 0,
  nfolds = 5,
  cores = 4,
  raw_data = FALSE
)
}
\arguments{
\item{tune_list}{This is a list object containing the values from which to develop the grid. For each of the hyperparameters
that can be tuned for (\code{num_hidden_layers}, \code{neurons}, \code{epochs}, \code{val_split}, \code{patience}, \code{learn_rate}, \code{num_basis},
\code{activation_choice}), the user inputs a set of values to try. Note that the combinations are found based on the number of
hidden layers. For example, if \code{num_hidden_layers} = 3 and \code{neurons} = c(8, 16), then the combinations will begin as
c(8, 8, 8), c(8, 8, 16), ..., c(16, 16, 16). Example provided below.}

\item{resp}{For scalar responses, this is a vector of the observed dependent variable. For functional responses,
this is a matrix where each row contains the basis coefficients defining the functional response (for each observation).}

\item{func_cov}{The form of this depends on whether the \code{raw_data} argument is true or not. If true, then this is
a list of k matrices. The dimensionality of the matrices should be the same (n x p) where n is the number of
observations and p is the number of longitudinal observations. If \code{raw_data} is false, then the input should be a tensor
with dimensionality b x n x k where b is the number of basis functions used to define the functional covariates, n is
the number of observations, and k is the number of functional covariates.}

\item{scalar_cov}{A matrix contained the multivariate information associated with the data set. This is all of your
non-longitudinal data.}

\item{basis_choice}{A vector of size k (the number of functional covariates) with either "fourier" or "bspline" as the inputs.
This is the choice for the basis functions used for the functional weight expansion. If you only specify one, with k > 1,
then the argument will repeat that choice for all k functional covariates.}

\item{domain_range}{List of size k. Each element of the list is a 2-dimensional vector containing the upper and lower
bounds of the k-th functional weight.}

\item{batch_size}{Size of the batch for stochastic gradient descent.}

\item{decay_rate}{A modification to the learning rate that decreases the learning rate as more and more learning
iterations are completed.}

\item{nfolds}{The number of folds to be used in the cross-validation process.}

\item{cores}{For the purpose of parallelization.}

\item{raw_data}{If True, then user does not need to create functional observations beforehand. The function will
internally take care of that pre-processing.}
}
\value{
The following are returned:

\code{Parameters} -- The final list of hyperparameter chosen by the tuning process.

\code{All_Information} -- A list object containing the errors for every combination in the grid. Each element of the list
corresponds to a different choice of number of hidden layers.

\code{Best_Per_Layer} -- An object that returns the best parameter combination for each choice of hidden layers.

\code{Grid_List} -- An object containing information about all combinations tried by the tuning process.
}
\description{
A convenience function for the user that implements a simple grid search for the purpose of tuning. For each combination
in the grid, a cross-validated error is calculated. The best combination is returned along with additional information.
This function only works for scalar responses.
}
\details{
No additional details for now.
}
\examples{
# libraries
library(fda)

# Loading data
data("daily")

# Obtaining response
total_prec = apply(daily$precav, 2, mean)

# Creating functional data
temp_data = array(dim = c(65, 35, 1))
tempbasis65  = create.fourier.basis(c(0,365), 65)
timepts = seq(1, 365, 1)
temp_fd = Data2fd(timepts, daily$tempav, tempbasis65)

# Data set up
temp_data[,,1] = temp_fd$coefs

# Creating grid
tune_list_weather = list(num_hidden_layers = c(2),
                         neurons = c(8, 16),
                         epochs = c(250),
                         val_split = c(0.2),
                         patience = c(15),
                         learn_rate = c(0.01, 0.1),
                         num_basis = c(7),
                         activation_choice = c("relu", "sigmoid"))

# Running Tuning
weather_tuned = fnn.tune(tune_list_weather,
                         total_prec,
                         temp_data,
                         basis_choice = c("fourier"),
                         domain_range = list(c(1, 24)),
                         nfolds = 2)

# Looking at results
weather_tuned

}
