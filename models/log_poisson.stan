data {
  int<lower=1> N;               // Number of individuals
  int<lower=1> P;               // Number of predictors
  int<lower=1> K;               // Number of time intervals
  matrix[N, P] X_data;          // Predictor matrix
  array[N, K] int<lower=0> quit;     // Event counts for each interval
  array[N, K] int<lower=0, upper=1> exposure; // At-risk indicators for each interval
}

parameters {
  vector[P] beta;               // Regression coefficients
  vector<lower=0>[K] lambda0;   // Baseline hazard for each interval
}

transformed parameters {
  matrix[N, K] lambda;          // Individual hazard rates
  matrix[N, K] mu;              // Expected counts
  
  
  vector[N] log_rel_risk;
  log_rel_risk = X_data * beta;  // N x P * P -> N x 1
    
  for (i in 1:N) {
    for (k in 1:K) {
      lambda[i,k] = exp(log_rel_risk[i]) * lambda0[k];
    }
  }
  
  // Calculate expected counts accounting for exposure
  for (i in 1:N) {
    for (k in 1:K) {
      mu[i,k] = exposure[i,k] * lambda[i,k];
    }
  }
}

model {
  lambda0 ~ gamma(0.01, 0.01);  
  beta ~ normal(0, 1);          
  
  // Likelihood
  for (i in 1:N) {
    for (k in 1:K) {
      if (exposure[i,k] == 1) {
        quit[i,k] ~ poisson(mu[i,k]);
      }
    }
  }
}

generated quantities {
  matrix[N, K] log_lik;
  
  for (i in 1:N) {
    for (k in 1:K) {
      if (exposure[i,k] == 1) {
        log_lik[i,k] = poisson_lpmf(quit[i,k] | mu[i,k]);
      } else {
        log_lik[i,k] = 0;
      }
    }
  }
}