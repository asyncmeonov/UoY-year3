data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
}
transformed data {
    vector[N] x2;
    x2 = x .* x;
} 
parameters {
  real a1;
  real a2;		 
  real b;
  real<lower=0> sigma;
}
model {
  a1 ~ normal(0,10);
  a2 ~ normal(0,10);    
  b ~ normal(0,10);
  sigma ~ cauchy(0,5);
//  for (n in 1:N)
//    y[n] ~ normal(a1*x[n] + a2*x2[n] + b, sigma);
  y ~ normal(a1*x + a2*x2 + b, sigma);
}
