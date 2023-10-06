import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from compute_scaling_experiments_helpers import get_cc_scalingEtoE, load_data, get_neuron, cvIsi, LVr
from compute_scaling_experiments_helpers import calculateFuncionalConnectivityCorrelations
from compute_scaling_experiments_helpers import lvr_from_isi, json_dump


number_of_simulated_2s_snippets = 6
first_snippet = 1

experiments = [
        'lower_etoi',
        'lichtman_chii2_smaller_g',
        'lichtman_chii2_different_time_consts',
        'lichtman_chii2_distributed_params',
        'lichtman_chii2_different_seed'
        ]

if len(sys.argv) > 1 and sys.argv[1] in experiments:
    simulation = sys.argv[1]
else:
    simulation = 'lichtman_chii2_different_seed'

if simulation == 'lichtman_chii2_different_seed':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_different_seed as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_stable_localEtoI1_lichtman_chiI2/'
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_different_seed'
    interesting_paths = [
            '07338a82a7f3331ed0309d0aefe2512b/',
            '0a5ef37413def0858a248422006c8d04/',
            '10acdb2ae9247b49d4eeea4d78f308b7/',
            '1d56199392523b0395e9e67a69fca968/',
            '33babd5e822097cd11e421d3df6065f5/',
            '3bd6c7565b57a129240be673cc9b1b5d/',
            '441263567203638a8250ec2593dec3ac/',
            '517f98422516bbe6cb324c5436e7f66f/',
            '6084159743c50b9a23e97a38d0b89daa/',
            '6e6de3f72cb1a380e4739e774ff732ca/',
            '8669f086a45d7fa5a253d6598f0f6663/',
            '912bda6cf185869da1db97f8afb5a2eb/',
            '9f692a9d98028fb6ef1fec9f56c42acb/',
            'a413e5bbaa628c51b03261c4e8c7ef1b/',
            'a8586e55554e8bb52a6b2bfd2cdb5423/',
            'b5816c8150c88676484652da56d5d603/',
            'b6ecf4c0943f997a0f270b90d863cced/',
            'f0a248e8f586b22fc3c00fa7ec02720b/',
            ]
    sim_path = 'de4934b8c7777751f7c516e2ad35f50a/'
    ana_path = 'd8057e7a966caf3eefdc9d234a0785d3/'
elif simulation == 'lower_etoi':
    from compute_scaling_experiments_helpers import state_lower_e_to_i as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_lower_e_to_i/'
    save_data_dir = 'simulated_data/scaling_experiment_lower_e_to_i'
    interesting_paths = [
            '12742e24273ad73c879fc396b575e2e9/',
            '2073b9923d7eb9b128c780a5d8ea9ba8/',
            '55b4bb7ec82018e02b0fd69517260c39/',
            '630a587e46e7a18beb16defe7ab53508/',
            '67120aba9cd1806fc6a47145baaf18d4/',
            '7284c68339a7743e55e6fb3da2d8955d/',
            '8fe1c1f7da65657072959fc060e3fce4/',
            '95be71aeac41f476e5baf643ec97aa52/',
            '96e2cc6124445ea07f925c1c1d3c3063/',
            'ac8d90fe0a8ab37e188f38066fd9bfd1/',
            'ece6d9902639ed3b4537a0ae9cbbc4da/',
            'f0d51a54183ce194c043f69f476f80aa/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = '099ed2eb40bf4285921a916d32bf5343/'
elif simulation == 'lichtman_chii2_smaller_g':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_smaller_g as state
    base_path = '/p/scratch/cjinb33/shimoura1/huvi_fraction_EI_stable_localEtoI1_lichtman_chiI2_smaller_g/'
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_smaller_g'
    interesting_paths = [
            '08e8618517e6e057db688d6257680da6/',
            '35022b8c38368b83fb671cfc1087d83f/',
            '59cc1c6418e57a0cac22d007f9e0467b/',
            '7d86a0be95d45469743dffa2b2809b8a/',
            'dbbc477edceb3323b0883c89f8d7d955/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = 'd8057e7a966caf3eefdc9d234a0785d3/'
elif simulation == 'lichtman_chii2_different_time_consts':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_different_time_consts as state
    base_path = '/p/scratch/cjinb33/shimoura1/huvi_fraction_EI_stable_localEtoI1_lichtman_chiI2_different_synaptic_time_constants/'
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_different_time_consts'
    interesting_paths = [
            '013292ec8fa6c9230d1c9518b71d19ad/',
            '01dde3bb7de102415c2c560fa65f9ee9/',
            '51d3f722fcea42fe722804af260521ad/',
            '615aa8683a7ed776eafc718065dcbd3d/',
            '6f68a4df981d21210c693cf61bc11bfe/',
            '938abb40e337c93bbb95087117e324a6/',
            '984f627232b71236276f8d5178c3baab/',
            'abab9e23cec17ce523088b36d5a4e6d8/',
            'ad2f36126638bf99c0418469d0bfebba/',
            'd9673c094027e9046fa6413950f32647/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = 'd8057e7a966caf3eefdc9d234a0785d3/'
elif simulation == 'lichtman_chii2_distributed_params':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_distributed_params as state
    base_path = '/p/scratch/cjinb33/shimoura1/huvi_fraction_EI_stable_localEtoI1_lichtman_chiI2_distributed_parameters/'
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_distributed_params'
    interesting_paths = [
            '2c60ac72127fa0089cf8c65a9f4aaf8e/',
            '349e4572253d7264d71a61b10dd4b7f3/',
            '38c92ec7b713e1af25169b09ea8b4155/',
            '3e2840d0e00cdbc0bdd11b22cb89c399/',
            '5cc5c45cfc49cd75d22c7a039732722e/',
            '5ea08803e2c6e3d97b8bf88b65d5885d/',
            '780e7ad49d1a1334217e385dd6d119d2/',
            '973f392e39de8e400a1ee0676371d35c/',
            '9a884e30b168fef077bacef05058921d/',
            'b2ccc6891543dbd89798aa0a0d9c10eb/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = 'd8057e7a966caf3eefdc9d234a0785d3/'

initial_transient = 2500.
exp_st_duration = 2000.
tmax = 12500.

LvR_min_spikes = 10
CV_min_spikes = 10
min_rate = .5

spike_path = base_path + '{}/' + sim_path + 'spikes.pkl'
popgids_path = base_path + '{}/' + sim_path + 'population_GIDs.pkl'
input_current_path = base_path + '{}/' + sim_path + ana_path + 'input_current.pkl'

area = 'caudalanteriorcingulate'

# =========================================================================
# ==============================================================================
# ==============================================================================
# Read in and process data
# =========================================================================
# ==============================================================================
# ==============================================================================

filename = '/p/project/cjinb33/jinb3330/gitordner/huvi/data/rutishauser/spikes/'
data = load_data(filename)
print(f'loaded {len(data)} neurons')
all_ids = range(len(data))

# ==============================================================================
# ==============================================================================
# Read in statistics
# ==============================================================================
# ==============================================================================

# ==============================================================================
# Experimental data
# ==============================================================================

exp_rates = []
exp_lvr = []
exp_cv = []
# Loop over all neurons
for id_ in all_ids:
    neuron = get_neuron(data, id_)
    stim_on = neuron['stim_on']
    # timepoint when the simulation is turned on, we want the spiketrain before
    # this. Thus disregard all spikes that follow this. This values is 2.
    baseline = neuron['baseline_stim_on']
    # Every neuron has several recordings. Loop over all these recordings
    for i, st in enumerate(stim_on):
        st = np.atleast_1d(st)
        st = st[st < baseline]
        exp_rates.append(st.size / baseline)
        if st.size >= LvR_min_spikes:
            isi = np.diff(st)
            if isi.min() > 0:
                exp_lvr.append(lvr_from_isi(isi))
        if st.size >= CV_min_spikes:
            isi = np.diff(st)
            if isi.min() > 0:
                exp_cv.append(np.std(isi) / np.mean(isi))
exp_rates = np.array(exp_rates)
exp_rates = exp_rates[exp_rates >= min_rate]
exp_lvr = np.array(exp_lvr)
exp_cv = np.array(exp_cv)

print('Exp', len(exp_rates))

# ==============================================================================
# Simulated data
# ==============================================================================

sim_rates = {}
sim_lvr = {}
sim_cv = {}

for type_, hash_ in state.items():
    spikes = pd.read_pickle(spike_path.format(hash_))
    popgids = pd.read_pickle(popgids_path.format(hash_))

    sim_rates[type_] = []
    sim_lvr[type_] = []
    sim_cv[type_] = []

    spks = spikes.loc[area].agg(list).agg(sum)

    t_max = 2.
    for j in range(first_snippet, number_of_simulated_2s_snippets):
        i = j - 1
        for st in spks:
            st = st[
                    (st > initial_transient + i * exp_st_duration)\
                            & (st < initial_transient + (i+1) * exp_st_duration)
                            ]
            sim_rates[type_].append(st.size / t_max)
            if st.size >= LvR_min_spikes:
                isi = np.diff(st)
                if isi.min() > 0:
                    sim_lvr[type_].append(lvr_from_isi(isi, tau_r=2.))
            if st.size >= CV_min_spikes:
                isi = np.diff(st)
                if isi.min() > 0:
                    sim_cv[type_].append(np.std(isi) / np.mean(isi))
    sim_rates[type_] = np.array(sim_rates[type_])
    sim_rates[type_] = sim_rates[type_][sim_rates[type_] >= min_rate]
    sim_lvr[type_] = np.array(sim_lvr[type_])
    sim_cv[type_] = np.array(sim_cv[type_])

    print(type_, len(sim_rates[type_]))

# ==============================================================================
# Similarity to spiking data expressed via the kolmogorov smirnov test for area
# ==============================================================================

# Filter experimental rates
exp_rates_ = [rate for rate in exp_rates if rate >= min_rate]
path = [os.path.join(base_path, x) for x in interesting_paths]
path_to_chi = {}
ks_rates_arr = {}
ks_cv_arr = {}
ks_lvr_arr = {}
for d in path:
    p_net = d
    p_sim = os.path.join(d, sim_path)
    p_ana = os.path.join(d, sim_path, ana_path)
    if os.path.isdir(p_net):
        cc_scaling = get_cc_scalingEtoE(p_net)
        spikes = pd.read_pickle(os.path.join(p_sim, 'spikes.pkl')).loc[area]
        spks = []
        sim_rates_list = []
        path_to_chi[cc_scaling] = d.split('/')[-2]
        for j in range(first_snippet, number_of_simulated_2s_snippets):
            i = j - 1
            r = (
                    spikes.apply(
                        lambda sts: np.array([st[(st > initial_transient + i * exp_st_duration) & (st < initial_transient + (i+1) * exp_st_duration)].size for st in sts])
                        ) * 1000. / exp_st_duration
                ).agg(list).agg(sum)
            sim_rates_ = [rate for rate in r if rate >= min_rate]
            sim_rates_list += sim_rates_

            spks_tmp = spikes.apply(
                    lambda sts: np.array([st[(st > initial_transient + i * exp_st_duration) & (st < initial_transient + (i+1) * exp_st_duration)] - i*exp_st_duration for st in sts])
                    ).agg(list).agg(sum)
            spks += spks_tmp

        cv, _ = cvIsi(
                spks,
                CV_min_spikes=CV_min_spikes
                )

        lv, _ = LVr(
                spks,
                2,
                LvR_min_spikes=LvR_min_spikes
                )

        ks_rates_arr[cc_scaling] = 1 - ks_2samp(exp_rates_, sim_rates_list)[0]
        ks_cv_arr[cc_scaling] = 1 - ks_2samp(exp_cv, cv)[0]
        ks_lvr_arr[cc_scaling] = 1 - ks_2samp(exp_lvr, lv)[0]

# ==============================================================================
# ==============================================================================
# Calculate correlation between fMRI and simulated fc
# ==============================================================================
# ==============================================================================

df_corr_lh = {}
df_corr_rh = {}
for use_corrcoeff in [True, False]:
    scale = []
    corr_left = []
    corr_right = []
    path = [os.path.join(base_path, x) for x in interesting_paths]
    for d in path:
        p_net = d
        p_ana = os.path.join(d, sim_path, ana_path)
        if os.path.isdir(p_net):
            chi = get_cc_scalingEtoE(p_net)
            scale.append(chi)
            exp__sim_syn__corr_lh, exp__sim_syn__corr_rh, exp_fc_mean_lh, exp_fc_array_rh = calculateFuncionalConnectivityCorrelations(p_ana, tmin=initial_transient, tmax=tmax, use_corrcoeff=use_corrcoeff)
            corr_left.append(exp__sim_syn__corr_lh)
            corr_right.append(exp__sim_syn__corr_rh)

    df_corr_lh[use_corrcoeff] = pd.Series(corr_left, index=scale).sort_index()
    df_corr_rh[use_corrcoeff] = pd.Series(corr_right, index=scale).sort_index()


# ==============================================================================
# Calculate fMRI correlation
# ==============================================================================

# Read values
simulated_fc = {}
for type_, hash_ in state.items():
    curr_in = pd.read_pickle(input_current_path.format(hash_))
    synaptic_currents = curr_in.T

    # Calculate simulated functional connectivity based on synaptic
    # currents
    df_sim_fc_syn = synaptic_currents[
            (synaptic_currents.index >= initial_transient) & (synaptic_currents.index <= tmax)
            ].corr()
    simulated_fc[type_] = df_sim_fc_syn

# ==============================================================================
# ==============================================================================
# Dump data
# ==============================================================================
# ==============================================================================

json_dump(path_to_chi, os.path.join(save_data_dir, 'path_to_chi.json'))

# ==============================================================================
# Dump data for plotting similarity plot
# ==============================================================================

json_dump(ks_rates_arr, os.path.join(save_data_dir, 'ks_rates_arr.json'))
json_dump(ks_cv_arr, os.path.join(save_data_dir, 'ks_cv_arr.json'))
json_dump(ks_lvr_arr, os.path.join(save_data_dir, 'ks_lvr_arr.json'))
for use_corrcoeff in [True, False]:
    df_corr_lh[use_corrcoeff].to_csv(os.path.join(save_data_dir, f'df_corr_lh_use_corrcoeff_{use_corrcoeff}.csv'))
    df_corr_rh[use_corrcoeff].to_csv(os.path.join(save_data_dir, f'df_corr_rh_use_corrcoeff_{use_corrcoeff}.csv'))

# ==============================================================================
# Dump data for distribution plots
# ==============================================================================

# Experiment
exp_rates.dump(os.path.join(save_data_dir, 'exp_rates.npy'))
exp_cv.dump(os.path.join(save_data_dir, 'exp_cv.npy'))
exp_lvr.dump(os.path.join(save_data_dir, 'exp_lvr.npy'))

# Simulation
for key in ['groundstate', 'metastable']:
    sim_rates[key].dump(os.path.join(save_data_dir, f'sim_rates_{key}.npy'))
    sim_cv[key].dump(os.path.join(save_data_dir, f'sim_cv_{key}.npy'))
    sim_lvr[key].dump(os.path.join(save_data_dir, f'sim_lvr_{key}.npy'))

# ==============================================================================
# Dump data for fMRI plot
# ==============================================================================

for key in ['groundstate', 'metastable']:
    simulated_fc[key].to_csv(os.path.join(save_data_dir, f'simulated_fc_{key}.csv'))
