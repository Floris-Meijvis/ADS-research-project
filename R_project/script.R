library(tidyverse)
library(ggplot2)

#we generate a dataset for 2 classes (0 and 1) with 1 predictors, 
#with x_1 contributing positively, assume x_1 falls in [-1,1]

#intercepts are very domineering.
a_0 = -0.5 #intercept class 1
a_1 = 1  #regression coef class 1
b_0 = 0 #intercept class 2
b_1 = 1.5 #regression coef class 2

N = 10000 #datapoints

set.seed(1234)
x = runif(n=N, min=-1, max=1)
P0 = (1)/(1 + exp(a_0 + a_1 * x) + exp(b_0 + b_1 * x))
P1 = (exp(a_0 + a_1 * x))/(1 + exp(a_0 + a_1 * x) + exp(b_0 + b_1 * x))
P2 = (exp(b_0 + b_1 * x))/(1 + exp(a_0 + a_1 * x) + exp(b_0 + b_1 * x))

y = rep(0,N)

for (i in 1:N) {
y[i] = sample(c(0,1,2), size = 1, replace = TRUE, prob = c(P0[i], P1[i], P2[i]))  #classes.
}
Data = data.frame(x, y)

# Function to compute log-likelihood
Log_Likelihood <- function(params, data) {
  x = data$x
  y = data$y
  a_0 = params[1]
  a_1 = params[2]
  b_0 = params[3]
  b_1 = params[4]
  
  # Compute probabilities
  log_denominator = log(1 + exp(a_0 + a_1 * x) + exp(b_0 + b_1 * x))
  log_p1 = a_0 + a_1 * x - log_denominator
  log_p2 = b_0 + b_1 * x - log_denominator
  log_p0 = -log_denominator
  
  # Select log-probabilities based on y values
  log_likelihoods = ifelse(y == 0, log_p0, ifelse(y == 1, log_p1, log_p2))
  
  return(-sum(log_likelihoods))  # Negative for minimization
}

Log_Lik = Log_Likelihood(c(a_0, a_1, b_0, b_1), Data)
# Optimization setup
initial_params = c(-0.5, 1, 0, 1.5)  # Initial guesses for (a_0, a_1, b_0, b_1)

opt_result <- optim(
  par = initial_params,            # Initial parameter values
  fn = Log_Likelihood,             # Function to minimize
  data = Data,                      # Pass the dataset separately
  method = "BFGS"                   # Optimization method
)

# Display results
print(opt_result$par)  # Optimized parameters
print(opt_result$value)  # Final log-likelihood value


#optim for MLE? uses BFGS


#Use BFGS procedure for MLE estimation https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
#but is less accurate than using newton-raphson with the hessian matrix.

#for MLE try to use a_1 = cb_1 with c \geq 0 (or c = delta^2) so  coefs have same sign? 

#Extend programm to 3 or more classes, then apply MLE with constraint of Alpha's having the same sign. 
#maybe after that expand to the more general condition.
