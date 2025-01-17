my_new_benchmark = mgab.generate_benchmark({ # we choose a few parameters ourselves
        'seed': 123,
        'output_dir' : 'my_new_benchmark', # specify new directory for the output files
        'output_force_override': True, # Override files, if necessary
        'num_series': 1, # Create only 3 time series for this benchmark
        'series_length': 50_000, # Only create time series of lenth 10k
        'num_anomalies' : 50, # Each time series contains 50 anomalies
        'noise' : 'rnd_uniform',# Add random uniform noise
        'noise_param' : (-0.01, 0.01), # range for random uniform noise
        'anomaly_window': 25,
        'max_sgmt': 50,
        'min_anomaly_distance' : 250, # Anomalies have to have a distance of at least 200
        'mg_tau' : 30, # use a larger value for tau
        'mg_beta': 0.25,
        'mg_ts_path_load' : None, # We do not have any pre-computed MG time series. So generate it with the DDE solver
        'mg_ts_dir_save' : "C:/Users/heol/Projects/MGAB/data/" # Save the generated MG time series of the DDE solver in the data directory. This
                                     # allows us, to reuse it again (e.g., if we want to change the number of anomalies)
     })