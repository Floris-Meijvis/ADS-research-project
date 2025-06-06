library(MASS)
library(VGAM)

set.seed(5)
n<-10000
k<-3 # number of predictors
mu<-rep(0,k)
Sigma<-diag(1,k)
X<-as.matrix(mvrnorm(n,mu,Sigma))
m<-4 # m+1 categories

a<-rnorm(m)
b<-exp(rnorm(k))

dn<-1
for (s in 1:m){dn<-dn+exp(a[s]+s*X%*%b)}
P<-matrix(,nrow=n,ncol=m+1)
P[,1]<-1/dn
for (s in 1:m){P[,s+1]<-exp(a[s]+s*X%*%b)/dn}

y<-c()
for (i in 1:n){y[i]<-sample(0:m,size=1,replace=T,prob=P[i,])}

########################################################################

Z<-matrix(,nrow=n,ncol=m)
for (i in 1:n){for (j in 1:m){if (y[i]==j){Z[i,j]<-1} else {Z[i,j]<-0}}}

ns<-colSums(Z)
d<-colSums(y*X)

logl<-function(y){dn<-1
for (s in 1:m){dn<-dn+exp(y[s]+s*X%*%y[(m+1):(m+k)])}
-ns%*%y[1:m]-d%*%y[(m+1):(m+k)]+sum(log(dn))}

#gradient function of loglik for adjacent categories
dl<-function(y){
  dn<-1
  nu<-0
  for (s in 1:m){dn<-dn+exp(y[s]+s*X%*%y[(m+1):(m+k)])
  nu<-nu+s*exp(y[s]+s*X%*%y[(m+1):(m+k)])}
  da<-c()
  for (s in 1:m){da[s]<--ns[s]+sum(exp(y[s]+s*X%*%y[(m+1):(m+k)])/dn)}
  return(c(da,-d+t(colSums(X*c(nu/dn)))))}

sv<-rep(1,m+k) # starting values

fit<-optim(sv,logl,dl,method="BFGS",lower=-Inf,upper=Inf,
           control=list(trace=TRUE,reltol=1e-80,abstol=1e-80,maxit=1000),hessian=TRUE)
fit
dl(fit$par)

cbind(fit$par,sqrt(diag(solve(fit$hessian))))  #gives estimates and std's of parameters

lrf<-vglm(y~X,family=acat(parallel=T)) # package VGAM
summary(lrf)

lrf0<-vglm(y~X,family=acat) # package VGAM
summary(lrf0)
coef(lrf0,matrix=TRUE)

lrf2<-vglm(y~X,family=multinomial(refLevel=1))
summary(lrf2)
