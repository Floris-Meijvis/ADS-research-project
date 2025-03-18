# Load necessary libraries
library(nnet) #for multinomial logistic regression
library(ggplot2) #for plots
library(tidyverse)  #for pipe operators

# Function to generate dataset
generate_multinomial_data <- function(N, k, p = 2) {
  # Generate predictor values X in [-1,1]
  X = matrix(runif(N * p, -1, 1), ncol = p)
  
  # Generate random coefficients for the multinomial logit model
  beta = matrix(rnorm((k - 1) * p), ncol = p)  # (k-1) x p coefficient matrix
  beta_signs = sample(c(-1, 1), p, replace = TRUE) # Assign sign per predictor
  
  # Ensure same sign for each predictor across classes using a for-loop
  for (j in 1:p) {
    beta[, j] = beta_signs[j] * abs(beta[, j])
  }
  
  # Generate intercept coefficients (alpha)
  alpha = rnorm(k - 1) # Intercepts for (k-1) classes
  
  # Compute linear predictors for each class (excluding reference class)
  eta = X %*% t(beta) + matrix(alpha, nrow = N, ncol = k - 1, byrow = TRUE) # N x (k-1)
  
  # Convert to probabilities using softmax transformation
  exp_eta = exp(cbind(0, eta)) # Add reference class with zero logit
  probs = exp_eta / rowSums(exp_eta) # Normalize
  
  # Sample class labels
  y = apply(probs, 1, function(prob) sample(1:k, 1, prob = prob))
  
  # Return dataset as a dataframe along with coefficients
  data = data.frame(y = as.factor(y), X)
  return(list(data = data, beta = beta, alpha = alpha))
}

# Example usage
set.seed(123) # For reproducibility
result = generate_multinomial_data(N = 100, k = 8, p = 4)
head(result$data)
result$beta
result$alpha

