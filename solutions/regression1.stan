data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
//  alpha ~ normal(-2,10);    
//  beta ~ normal(3.5,10);
  alpha ~ normal(-2,0.1);    
  beta ~ normal(3.5,0.1);
  sigma ~ cauchy(0,5);
  for (n in 1:N)
    y[n] ~ normal(alpha + beta * x[n], sigma);
}
