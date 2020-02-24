data {
    int<lower=0> N;
    int<lower=0> r[N];
    int<lower=0> n[N];
    vector[N] x1; 
    vector[N] x2; 
}

//remove this interaction term: trash alpha12 and their product in the r allocation
transformed data {
    vector[N] x1x2;
    x1x2 = x1 .* x2;
} 

parameters {
    real alpha0;
    real<upper=0> alpha1;
    real alpha12;
    real<lower=0> alpha2;
}

model {
    //comment out if you want stan to infer the priors
//    alpha0 ~ normal(0.0,1.0E3);
//    alpha1 ~ normal(0.0,1.0E3);
//    alpha2 ~ normal(0.0,1.0E3);
//    alpha12 ~ normal(0.0,1.0E3);

   r ~ binomial_logit(n, alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2);
   //r ~ binomial_logit(n, alpha0 + alpha1 * x1 + alpha2 * x2);

}