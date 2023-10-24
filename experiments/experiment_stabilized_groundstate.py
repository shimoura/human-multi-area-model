# Overwrite values of python/default_net_params.py
net_params = {}

# Overwrite values of python/default_sim_params.py
sim_params = {
    't_sim': 12500.0,
    'master_seed': 2106
}

# Parameters for the analysis
ana_params = {
    'plotRasterArea': {
        'fraction': 0.05,
        'low': 2500,
        'high': 12500
    },
    'functconn_corr': {
        'exclude_diagonal': False
    }
}
