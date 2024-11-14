data {
    int<lower=1> N;                   // Total number of observations
    int<lower=1> N_f;                 // Number of observations for group 'f' 
    int<lower=1> N_m;                 // Number of observations for group 'm' 
    int<lower=1> P;                   // Number of predictors
    int<lower=1> K;                   // Number of time points or intervals
    int<lower=1> G;                   // Number of groups (by gender)
    int<lower=0> F;                   // Number of unique frailty fields or groups

    array[2] real optm_params;        // Optimized parameters, e.g., hyperparameters for frailty

    matrix[N_f, P] X_data_f;          // Predictor matrix for group 'f'
    matrix[N_m, P] X_data_m;          // Predictor matrix for group 'm'

    array[N, K] int<lower=0, upper=1> quit;       // Binary outcome matrix indicating quitting events
    matrix<lower=0, upper=1>[N, K] exposure;      // Binary matrix indicating exposure at each interval

    array[N] int<lower=1, upper=F> frailty_idx;   // Index for frailty groups for each observation
}

parameters {
    vector[P] beta;                   // Coefficients for predictors
    matrix<lower=0>[K, G] lambda0;    // Baseline rate matrix for each group and interval
    real<lower=0> sigma_frailty;      // Standard deviation for frailty effect
    real mu_frailty;                  // Mean for frailty effect
    vector<lower=0>[F] frailty;       // Frailty effect for each field
}

transformed parameters {
    matrix[N_m, K] lambda_m;
    matrix[N_f, K] lambda_f;
    matrix[N, K] lambda_;
    matrix[N, K] mu; 

    vector[N_m] tmp_m;
    vector[N_f] tmp_f;

    tmp_m = X_data_m * beta;
    tmp_f = X_data_f * beta;

    for (n in 1:N_f) {
        for (k in 1:K) {
            lambda_f[n, k] = exp(tmp_f[n]) * lambda0[k, 2];
        }
    }

    for (n in 1:N_m) {
        for (k in 1:K) {
            lambda_m[n, k] = exp(tmp_m[n]) * lambda0[k, 1];
        }
    }

    for (n in 1:N) {
        lambda_[n, ] = frailty[frailty_idx[n]] * ((n <= N_f) ? lambda_f[n, ] : lambda_m[n - N_f, ]);
    }

    mu = exposure .* lambda_;
}

model {
    // Priors
    beta ~ normal(0, 1);                        
    sigma_frailty ~ normal(optm_params[1], 1);  
    mu_frailty ~ normal(optm_params[2], 1);     
    frailty ~ gamma(mu_frailty, sigma_frailty);

    for (g in 1:G) {
        lambda0[, g] ~ gamma(0.01, 0.01);       // Baseline rate prior for each group
    }

    // Likelihood
    for (n in 1:N) {
        for (k in 1:K) {
            if (exposure[n, k] == 1) {
                quit[n, k] ~ poisson(mu[n, k]);  // Poisson likelihood for quitting events
            }
        }
    }
}

generated quantities {
    matrix[N, K] log_lik;  

    for (n in 1:N) {
        for (k in 1:K) {
            if (exposure[n, k] == 1) {
                log_lik[n, k] = poisson_lpmf(quit[n, k] | mu[n, k]);
            } else {
                log_lik[n, k] = 0;
            }
        }
    }
}
