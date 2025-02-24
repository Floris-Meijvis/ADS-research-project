library(tidyverse)

#we generate a dataset for 2 classes (0 and 1) with 1 predictors, 
#with x_1 contributing positively, assume x_1 falls in [-1,1]

a = -0.5 #intercept class 1
b = 1  #regression coef class 1
N = 10000 #datapoints

set.seed(1234)
x = runif(n=N, min=-1, max=1)
P0 = (1)/(1 + exp(a + b * x))
P1 = (exp(a + b * x))/(1 + exp(a + b * x))

help = runif(n=N, min=0, max=1) #random number for class assignment
y = rep(0,N)  #classes.

#we use the previous probabilities to assign classes to our data points
Data = data.frame(x, P0, P1, help, y) %>%
  mutate(y = ifelse(help > P0, 1, 0 )) %>%  
  select(x, y)