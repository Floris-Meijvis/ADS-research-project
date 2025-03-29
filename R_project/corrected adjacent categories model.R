# Load necessary libraries and set a seet
library(nnet) #for multinomial logistic regression
library(ggplot2) #for plots
library(tidyverse)  #for pipe operators
library(stats)

set.seed(123) # For reproducibility

#setup of parameters
N = 1000
k = 5
p = 3


#functions to use for data generation

# Function to generate dataset according to adjacent categories model
generate_adjacent_categories_data = function(N, k, p = 2) {
  # Generate predictor matrix X
  X = matrix(runif(N * p, -1, 1), nrow = N, ncol = p)
  
  # Generate coefficients for each predictor (only one set of betas)
  alpha = sort(runif(k - 1, -1, 1))  # Intercepts
  beta = rnorm(p)  # Single set of regression coefficients for all categories
  
  # Compute cumulative log-odds
  log_odds = matrix(NA, nrow = N, ncol = k - 1)
  for (j in 1:(k - 1)) {
    log_odds[, j] = alpha[j] + j * X %*% beta
  }
  
  # Convert log odds to probabilities
  probs = exp(log_odds)
  probs = cbind(1, probs)  # Include baseline category (category 1)
  probs = probs / rowSums(probs)  # Normalize to sum to 1
  
  # Generate categorical outcomes by sampling based on probabilities
  Y = apply(probs, 1, function(p) sample(0:(k-1), size = 1, prob = p))
  
  # Return as a data frame
  data = data.frame(Y = as.factor(Y), X)
  colnames(data) = c("Y", paste("X", 1:p, sep = ""))
  
  return(list(data = data, beta = beta, alpha = alpha))
}

#generating data

#adjacent_categories dataset
data_adjacent = generate_adjacent_categories_data(N, k, p)
head(data_adjacent$data)
data_adjacent$beta
data_adjacent$alpha

logLikAdjacent = function(par, data) {
  #INPUT:
  #par: flattened parameter matrix, see example below for correct creation
  #data: dataset to calculate likelihood for, first column should equal class and be called Y
  
  # Extract dimensions
  num_classes = length(unique(data$Y)) - 1  # Excluding reference class
  num_predictors = ncol(data) - 1  # Excluding response variable
  
  # Reconstruct matrix parameters from par vector
  intercepts = par[1:num_classes]  # First few elements are intercepts
#  coefs = matrix(par[(num_classes + 1):length(par)], nrow = num_classes, ncol = num_predictors)
  coefs = par[(num_classes + 1):length(par)] # this one is new
  
  # Extract variables
  X = as.matrix(data[, -1])  # Convert to matrix
  Y = as.integer(data$Y) - 1  # Convert factor to integer (-1 to align indices)
  
  D = matrix(,nrow=length(Y),ncol=num_classes) # this one is new
  for (i in 1:length(Y)){for (j in 1:num_classes){if (Y[i]==j){D[i,j]=1} else {D[i,j]=0}}} # this one is new

  # Compute the linear predictors for all classes
#  eta = sweep(Y * X %*% t(coefs), 2, intercepts, "+")
  eta = Y * X %*% coefs + D %*% intercepts # this one is new
  ETA = matrix(rep(1:(k-1),length(Y)),nrow=length(Y),byrow=T) * c(X %*% coefs) + matrix(rep(intercepts,length(Y)),nrow=length(Y),byrow=T) # this one is new
  
  # Compute the sum of exponentials for the denominator
#  sumExp = rowSums(exp(eta)) + 1  # adding 1 for reference class Y=0
   sumExp = rowSums(exp(ETA)) + 1 # this one is new
  
  loglik = numeric(length(Y))  #Initialize log-likelihood vector
  
  # Compute log-likelihood per observation
  for (i in seq_along(Y)) { loglik[i]<-eta[i]-log(sumExp[i]) # this one is new
    #if (Y[i] == 0) {
    #  loglik[i] <- -log(sumExp[i])  # Reference class
    #} else {
    #  loglik[i] <- eta[i, Y[i]] - log(sumExp[i])
    #}
  }
  
  return(-sum(loglik))  # Negative log-likelihood for minimization
}

#example use
params_adjacent = c(data_adjacent$alpha, data_adjacent$beta)
logLikAdjacenttest = logLikAdjacent(params_adjacent, data_adjacent$data)

# Optimisation procedure for adjacent category model on adjacent categories data

#initial parameters, good guess is important
initial_params = c(rep(data_adjacent$alpha),  # Intercepts
                   data_adjacent$beta)   # Coefficients

opt_result = optim(
  par = initial_params,          # Flattened initial parameters
  fn = logLikAdjacent,           # Function to minimize
  data = data_adjacent$data,     # Pass the dataset separately
  method = "BFGS"                # Optimization method
)

#Reconstruction of original datashape and coefficients
optimized_params_adjacent = opt_result$par
optimized_intercepts_adjacent = optimized_params_adjacent[1:(k-1)]
optimized_coefs_adjacent = matrix(optimized_params_adjacent[k:length(optimized_params_adjacent)], 
                         nrow = k-1, 
                         ncol = p)

# bias adjacent categories model
errorcoefs = abs(data_adjacent$beta - optimized_coefs_adjacent)
errorintercept = abs(data_adjacent$alpha - optimized_intercepts_adjacent)
error_adjacent = c(errorintercept, errorcoefs)
print(paste0("average abs bias in parameter estimates when fitting adjacent categories model to adjacent categories dataset: ", mean(error_adjacent)))
