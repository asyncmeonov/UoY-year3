//A binary logistic regression model for the 30 features in the breast cancer data
data {
  //train
  int<lower=0> N;
  matrix[N,30] x;
  int<lower=0,upper=1> y[N];

  //test
  int<lower=0> N_new;
  matrix[N_new,30] x_new;

}
parameters {
  real alpha;
  vector[30] beta;
}


model {
    for (n in 1:N)
      y[n] ~ bernoulli_logit(alpha + dot_product(beta, x[n]));
}

generated quantities{
  vector[N_new] y_new;
  for (n in 1:N_new)
    y_new[n] = bernoulli_logit_rng(alpha + dot_product(beta, x_new[n]));
  
}

// data {
//    int<lower=0> N;  // size of training data
//    int<lower=0> M;  // size of test data	 
//    matrix[N,30] x[M+N];	     
//    int<lower=0,upper=1> klass[N];
//    vector<lower=0>[2] alpha0;
//    vector<lower=0>[2] alpha1;
//    vector<lower=0>[2] alpha2;
// }
// parameters {
//    simplex[2] theta0;    // class distribution
//    simplex[2] theta1[2]; // class conditional distributions for x1
//    simplex[2] theta2[2]; // class conditional distributions for x2
// }
// model {
//    theta0 ~ dirichlet(alpha0);
//    for (k in 1:2) {		 
//       theta1[k] ~ dirichlet(alpha1); // using same prior for both class values
//       theta2[k] ~ dirichlet(alpha2); // using same prior for both class values
//       }
//    for (n in 1:N) {
//       klass[n] ~ categorical(theta0);
//       x1[n] ~ categorical(theta1[klass[n]]);
//       x2[n] ~ categorical(theta2[klass[n]]);
//       }
// }
// generated quantities {
//    int<lower=1,upper=2> klass_pred[M];
//    for (n in 1:M) {
//    real p1;
//    real p2;
//    vector[2] p; 
//    p1 = theta0[1]*theta1[1,x1[N+n]]*theta2[1,x2[N+n]];
//    p2 = theta0[2]*theta1[2,x1[N+n]]*theta2[2,x2[N+n]];
//    p[1] = p1/(p1+p2);
//    p[2] = p2/(p1+p2);		     
//    klass_pred[n] = categorical_rng(p);
//   }
// }