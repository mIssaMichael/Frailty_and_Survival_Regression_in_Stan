data {
    int<lower=1> N;  // Number of observations
    int<lower=1> P;  // Number of predictors
    matrix[N, P] X_data;  // Predictor matrix
    vector[N] y;  // Time to event
    array[N] int<lower=0, upper=1> cens;  // Censoring indicator (1 = censored)
    int<lower=0, upper=1> is_weibull;  // Model type indicator (1 = Weibull, 0 = Logistic)
}

parameters {
    vector[P] beta;  // Regression coefficients
    real mu;  // Intercept
    real<lower=0> s;  // Scale parameter
}

transformed parameters {
    vector[N] eta = X_data * beta;
    vector[N] reg;
    
    if (is_weibull == 1) {
        reg = exp(-(mu + eta) / s);
    } else {
        reg = mu + eta;
    }
}

model {
    // Priors
    beta ~ normal(0, 1);
    mu ~ normal(0, 1);
    s ~ normal(0, 5);  
    
    if (is_weibull == 1) {
        // Weibull model with raw y values
        for (n in 1:N) {
            if (cens[n] == 0) {
                target += weibull_lpdf(y[n] | s, reg[n]);
            } else {
                target += weibull_lccdf(y[n] | s, reg[n]);
            }
        }
    } else {
        // Logistic model with log(y) values
        for (n in 1:N) {
            if (cens[n] == 0) {
                target += logistic_lpdf(y[n] | reg[n], s);
            } else {
                target += logistic_lccdf(y[n] | reg[n], s);
            }
        }
    }
}

generated quantities {
    vector[N] log_lik;
    vector[N] y_pred;
    
    for (n in 1:N) {
        if (is_weibull == 1) {
            if (cens[n] == 0) {
                log_lik[n] = weibull_lpdf(y[n] | s, reg[n]);
            } else {
                log_lik[n] = weibull_lccdf(y[n] | s, reg[n]);
            }
        } else {
            if (cens[n] == 0) {
                log_lik[n] = logistic_lpdf(y[n] | reg[n], s);
            } else {
                log_lik[n] = logistic_lccdf(y[n] | reg[n], s);
            }
        }
    }
}