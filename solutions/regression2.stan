data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x1;
  vector[N] x2;			     
  vector[N] x3;			     
}
parameters {
  real a;
  real b;
  real c;		 
  real d;
  real<lower=0> sigma;
}
model {
  a ~ normal(0,10);    
  b ~ normal(0,10);
  c ~ normal(0,10);
  d ~ normal(0,10);
  sigma ~ cauchy(0,5);
  for (n in 1:N)
    y[n] ~ normal(a * x1[n] + b * x2[n] + c * x3[n] + d, sigma);
}
