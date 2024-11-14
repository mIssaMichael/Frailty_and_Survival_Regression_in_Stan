functions {
    /**
     * Function to compute Kaplan-Meier estimates and confidence intervals.
     * @param N Number of observations
     * @param durations Vector of observed durations
     * @param event_observed Vector of binary indicators for event (1 if event occurred, 0 if censored)
     * @return A 3-column matrix: [times, km_estimate, lower_ci, upper_ci]
     */
    matrix km_estimate_func(int N, vector durations, vector event_observed) {
        vector[N] km_estimate;
        vector[N] lower_ci;
        vector[N] upper_ci;
        vector[N] times;
        
        // Sort the durations and event_observed
        array[N] int sorted_index = sort_indices_asc(durations);
        vector[N] sorted_durations = durations[sorted_index];
        vector[N] sorted_event_observed = event_observed[sorted_index];

        times = sorted_durations; 

        // Initialize the first estimate
        km_estimate[1] = 1.0;

        // Calculate Kaplan-Meier estimates
        for (i in 2:N) {
            if (sorted_event_observed[i - 1] == 1) {
                km_estimate[i] = km_estimate[i - 1] * (1 - sorted_event_observed[i - 1] / (N - (i - 1)));
            } else {
                km_estimate[i] = km_estimate[i - 1];
            }
        }

        // Calculate confidence intervals
        for (i in 1:N) {
            if (sorted_event_observed[i] > 0) {
                real se = sqrt((sorted_event_observed[i] / 
                                ((N - (i - 1)) * (N - (i - 1)))) * 
                               (N - (i - 1) - sorted_event_observed[i]) / 
                               (N - (i - 1) - 1));
                lower_ci[i] = km_estimate[i] - 1.96 * se;
                upper_ci[i] = km_estimate[i] + 1.96 * se;

                // Apply boundary corrections
                lower_ci[i] = lower_ci[i] < 0 ? 0 : lower_ci[i];
                upper_ci[i] = upper_ci[i] > 1 ? 1 : upper_ci[i];
            } else {
                lower_ci[i] = km_estimate[i];
                upper_ci[i] = km_estimate[i];
            }
        }

        // Return a matrix with times, estimates, and confidence intervals
        matrix[N, 4] km_output;
        km_output[, 1] = times;
        km_output[, 2] = km_estimate;
        km_output[, 3] = lower_ci;
        km_output[, 4] = upper_ci;

        return km_output;
    }
}

data {
    int<lower=0> N;                // Number of observations
    vector[N] durations;           // Durations
    vector[N] event_observed;      // Binary Event observed 
}

transformed data {
    // Get sorted indices based on durations
    array[N] int sorted_index = sort_indices_asc(durations);

    // Sort durations and event_observed using the sorted indices
    vector[N] sorted_durations = durations[sorted_index];
    vector[N] sorted_event_observed = event_observed[sorted_index];
}

generated quantities {
    matrix[N, 4] km_results;
    
    km_results = km_estimate_func(N, durations, event_observed);
    
}





