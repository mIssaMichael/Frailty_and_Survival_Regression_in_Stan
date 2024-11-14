data {
  int<lower=1> N;                // Number of data points
  array[N] int<lower=0> Z;             // Upper bound for summation (Z_i(n))
  array[N] real<lower=0> xi;           // Parameter for the negative binomial (mean)
  real<lower=0> phi;             // Dispersion parameter for the negative binomial
  array[N] int<lower=0> v;             // Number of trials for the binomial
  array[N] int<lower=0> v_plus;        // Number of successes for the binomial
}

parameters {
  real<lower=0> rho;             // Parameter scaling factor (e.g., mean multiplier)
}

model {
  for (n in 1:N) {
    // Array to store the likelihood components
    vector[Z[n] + 1] l;
    
    // Compute the denominator for truncation of the negative binomial
    real log_cdf_upper = neg_binomial_2_lcdf(Z[n] | xi[n], phi);
    
    // Iterate over all possible values of z (from 0 to Z_i(n))
    for (z in 0:Z[n]) {
      // Log-likelihoods for negative binomial and binomial
      real log_nb = neg_binomial_2_lpmf(z | xi[n], phi) - log_cdf_upper;
      real log_bin = binomial_lpmf(v_plus[n] | v[n], z / (1.0 * Z[n]));
      
      // Combine the likelihoods
      l[z + 1] = exp(log_nb + log_bin);
    }
    
    // Increment the target with the log of the sum of likelihoods
    target += log(sum(l));
  }
}
