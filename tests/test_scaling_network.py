import sys
import os
import numpy as np
import pytest

# Add the source directory to the system path
sys.path.append('./src/')

from data_preprocessing.cytoarchitecture import NeuronNumbers
from data_preprocessing.connectivity import SynapseNumbers
from network import Network, networkDictFromDump
from default_net_params import params as net_params

# Constants
SCALING_FACTOR = 0.005

@pytest.fixture
def setup_network():
    # Update network parameters with scaling factors
    net_params.update({
        'N_scaling': SCALING_FACTOR,
        'K_scaling': SCALING_FACTOR,
        'fullscale_rates': './simulated_data/base_theory_rates.pkl'
    })

    # Paths
    base_path = os.getcwd()
    outpath = net_params['outpath']

    # Initialize NeuronNumbers and SynapseNumbers classes
    neuron_numbers = NeuronNumbers(
        surface_area=net_params['surface_area'],
        **net_params['cytoarchitecture_params']
    )
    synapse_numbers = SynapseNumbers(
        NN=neuron_numbers,
        **net_params['predictive_connectomic_params']
    )

    # Get full-scale neuron and synapse numbers
    fullscale_NN_SN = {
        'NN': neuron_numbers.getNeuronNumbers(),
        'SN': synapse_numbers.getSynapseNumbers()
    }

    # Create and dump the network
    network = Network(neuron_numbers, synapse_numbers, net_params)
    network.dump(outpath)

    # Get network hash and load network dictionary
    net_hash = network.getHash()
    net_folder = os.path.join(outpath, net_hash)
    net_dict = networkDictFromDump(net_folder)

    return fullscale_NN_SN, net_dict

def test_scaling_factors(setup_network, capsys):
    fullscale_NN_SN, net_dict = setup_network

    # Sort full-scale neuron and synapse numbers
    full_scale_neurons = fullscale_NN_SN['NN'].sort_index()
    full_scale_synapses = fullscale_NN_SN['SN'].sort_index(axis=0).sort_index(axis=1)

    # Sort network dictionary neuron and synapse numbers
    net_dict['neuron_numbers'] = net_dict['neuron_numbers'].sort_index()
    net_dict['synapses_internal'] = net_dict['synapses_internal'].sort_index(axis=0).sort_index(axis=1)

    # Validate the downscaling factors
    expected_neurons = np.round(full_scale_neurons * SCALING_FACTOR).astype(int)
    expected_synapses = np.round(full_scale_synapses * net_params['N_scaling'] * net_params['K_scaling']).astype(int)

    assert np.all(net_dict['neuron_numbers'] == expected_neurons), "Neuron numbers do not match expected downscaled values."
    assert np.all(net_dict['synapses_internal'] == expected_synapses), "Synapse numbers do not match expected downscaled values."

    print("Success: Neuron and synapse numbers match the expected downscaled values.")
    
    # Capture and print the output
    captured = capsys.readouterr()
    print(captured.out)

if __name__ == "__main__":
    pytest.main([__file__])