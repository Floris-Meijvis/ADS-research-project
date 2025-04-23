library(MASS)
library(VGAM)

set.seed(5)
n<-10000
k<-5 # number of predictors
mu<-rep(0,k)
Sigma<-diag(1,k)
X<-as.matrix(mvrnorm(n,mu,Sigma))
m<-3 # m+1 categories

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

#ordinal model
lrf<-vglm(y~X,family=acat(parallel=T)) # package VGAM
summary(lrf)

#adjacent categories (nominal // reparametrised)
lrf0<-vglm(y~X,family=acat) # package VGAM
summary(lrf0)
coef(lrf0,matrix=TRUE)

#usual nominal LR model
lrf2<-vglm(y~X,family=multinomial(refLevel=1))
summary(lrf2)
