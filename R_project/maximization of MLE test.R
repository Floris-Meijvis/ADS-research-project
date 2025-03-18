library(tidyverse)
library(ggplot2)

#we generate a dataset for 3 classes (0, 1, 2) with 1 predictors, 
#with x_1 contributing positively, assume x_1 falls in [-1,1]

#intercepts values are very important.
a_0 = -0.5 #intercept class 1
a_1 = 1  #regression coef class 1 predictor 1
a_2 = -0.5  #regression coef class 1 predictor 2
b_0 = 0 #intercept class 2
b_1 = 1.5 #regression coef class 2 predictor 1
b_2 = -1 #regression coef class 2 predictor 2

N = 10000 #datapoints

set.seed(1234)
x_1 = runif(n=N, min=-1, max=1)
x_2 = runif(n = N, min = -1, max = 1)
P0 = (1)/(1 + exp(a_0 + a_1 * x_1 + a_2 * x_2) + exp(b_0 + b_1 * x_1 + b_2 * x_2))
P1 = (exp(a_0 + a_1 * x_1 + a_2 * x_2))/(1 + exp(a_0 + a_1 * x_1 + a_2 * x_2) + exp(b_0 + b_1 * x_1 + b_2 * x_2))
P2 = (exp(b_0 + b_1 * x_1 + b_2 * x_2))/(1 + exp(a_0 + a_1 * x_1 + a_2 * x_2) + exp(b_0 + b_1 * x_1 + b_2 * x_2))

y = rep(0,N)

for (i in 1:N) {
y[i] = sample(c(0,1,2), size = 1, replace = TRUE, prob = c(P0[i], P1[i], P2[i]))  #classes.
}
Data = data.frame(x_1, x_2, y)

# Function to compute log-likelihood
Log_Likelihood <- function(params, data) {
  x_1 = data$x_1
  x_2 = data$x_2
  y = data$y
  a_0 = params[1]
  a_1 = params[2]
  a_2 = params[3]
  b_0 = params[4]
  b_1 = params[5]^2 * params[2] # this ensures both have same sign
  b_2 = params[6]^2 * params[3]
  
  # Compute probabilities
  log_denominator = log(1 + exp(a_0 + a_1 * x_1 + a_2 * x_2) + exp(b_0 + b_1 * x_1 + b_2 * x_2))
  log_p1 = a_0 + a_1 * x_1 + a_2 * x_2 - log_denominator
  log_p2 = b_0 + b_1 * x_1 + b_2 * x_2 - log_denominator
  log_p0 = -log_denominator
  
  # Select log-probabilities based on y values
  log_likelihoods = ifelse(y == 0, log_p0, ifelse(y == 1, log_p1, log_p2))
  
  return(-sum(log_likelihoods))  # Negative for maximization
}

Log_Lik = Log_Likelihood(c(a_0, a_1, a_2, b_0, b_1, b_2), Data)
# Optimization setup
initial_params = c(-0.5, 1, -0.5, 0, 1.5, -1)  # Initial guesses for params

opt_result <- optim(
  par = initial_params,            # Initial parameter values
  fn = Log_Likelihood,             # Function to minimize
  data = Data,                      # Pass the dataset separately
  method = "BFGS"                   # Optimization method
)

print(opt_result$value)  # Final log-likelihood value
#Uses BFGS procedure for MLE estimation https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
#but is less accurate than using newton-raphson with the hessian matrix

#predicted parameters
a_0_pred = opt_result$par[1]
a_1_pred = opt_result$par[2]
a_2_pred = opt_result$par[3]
b_0_pred = opt_result$par[4]
b_1_pred = opt_result$par[5]^2 * opt_result$par[2]
b_2_pred = opt_result$par[6]^2 * opt_result$par[3]

#for simulation study: set up the above programm to setup sample size, #predictors and #classes beforehand. 
#that way you can compare between models at different values.

.
