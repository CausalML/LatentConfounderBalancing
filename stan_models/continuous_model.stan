data {
  int<lower=1> N; // number of data points
  int<lower=1> Z; // dimensionality of Z
  int<lower=1> X; // dimensionality of X
  int<lower=1> T; // number of treatments
  matrix[X, Z] R; // mean-x transformation matrix
  vector[X] R_0; // mean-x intercept
  vector[X] std_x; // std-x
  matrix[T, Z] P; // logit t transoformation matrix
  vector[T] P_0; // logit t intercept

  int<lower=1, upper=T> t[N]; // treatments
  vector[X] x[N]; // covariates 
}

parameters {
  vector[Z] z[N]; // z values
}

transformed parameters {
  vector[X] mu_x[N];
  vector[T] logit_t[N];
  for (n in 1:N)
  {
    mu_x[n] = R * z[n] + R_0;
    logit_t[n] = P * z[n] + P_0;
  }
}

model {
  for (n in 1:N) {
    z[n] ~ normal(0, 1);
    t[n] ~ categorical_logit(logit_t[n]);
    x[n] ~ normal(mu_x[n], std_x);
  }
}


