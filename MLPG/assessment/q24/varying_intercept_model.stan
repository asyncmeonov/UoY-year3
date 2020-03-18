data {
    int<lower=0> N; // total number of houses
    int<lower = 1, upper = 2> L[N];
    vector[N] A; // predictor Age
    vector[N] S; // predictor Size
    vector[N] P; // outcome vector, Price
}
parameters {
    vector[2] alpha; // intercepts for the two localities
    real beta_A;
    real<lower=0> beta_S; 
    real<lower=0> sigma_a; // intercept noise
    real<lower=0> sigma_y; // general noise
}
transformed parameters {
    vector[N] y_hat;
    for (i in 1:N)
      y_hat[i] = alpha[L[i]] + A[i]*beta_A + S[i]*beta_S;
}

model {
    alpha ~ normal(0, sigma_a); // mean 0 and variance is learned
    P ~ normal(y_hat, sigma_y); // likelihood
}