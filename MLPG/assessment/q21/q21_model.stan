data {
    int<lower=0> N; // number of data items
    vector[N] A; // predictor Age
    vector[N] L; // predictor Locality
    vector[N] S; // predictor Size
    vector[N] P; // outcome vector
}
parameters {
    real alpha; // intercept
    real beta_L;
    real beta_A;
    real beta_S; 
    real<lower=0> sigma; // error scale
}
model {
    P ~ normal(A * beta_A + S * beta_S + L * beta_L + alpha, sigma); // likelihood
}
