# Overwrite values of python/default_net_params.py
net_params = {
    'neuron_model_E': 'mat2_psc_exp',
    'neuron_params_E': {
        'E_L': -70.0,
        'omega': -50.0,
        'C_m': 160.0,
        'tau_m': 10.0,
        'tau_syn_ex': 0.5,
        'tau_syn_in': 0.5,
        't_ref': 2.0,
        'tau_1': 10.0,
        'alpha_1': 20.0,
        'tau_2': 200.0,
        'alpha_2': 2.0
    }
}

# Overwrite values of python/default_sim_params.py
sim_params = {
    't_sim': 1000.0,
    'sim_resolution': 0.1,
}

# Parameters for the analysis
ana_params = {
    'rate_histogram_binsize': 1.
}
