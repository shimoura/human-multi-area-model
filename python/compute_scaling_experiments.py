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
        'normal',
        'lower_etoi',
        'normal_longer_time',
        'distributed_params',
        'different_synaptic_time_constants',
        'smaller_g',
        'lichtman',
        'lichtman_chii2',
        'lichtman_chii2_smaller_g',
        'lichtman_chii2_different_time_consts',
        'lichtman_chii2_distributed_params',
        'lichtman_chii2_different_seed'
        ]

if len(sys.argv) > 1 and sys.argv[1] in experiments:
    simulation = sys.argv[1]
else:
    simulation = 'normal'

if simulation == 'normal':
    from compute_scaling_experiments_helpers import state_scaling as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_initial_values/'
    save_data_dir = 'data/scaling_experiment'
    interesting_paths = [
            '028fe4ee6cb64e07220df0d9fa52bfcc/',
            '09e43f2aaa43abb4fca80d2c83c45009/',
            '1637dcc755d2aab4fcb77ed000be5108/',
            '170aec8fbe72cffabfb9dfad48f1cb5e/',
            '227b4854e305a8927999d82e32b8d83b/',
            '34876107e16c952b639b4b115120d161/',
            '3f2b943ce302c7638873b533ca30e884/',
            '5d40feb84afec81f2116b1fd590b0837/',
            '6c0ff4c3e733cf0fd1dc9485545cf027/',
            '6cefaea97aa6fecdce26a56ef6f6be00/',
            '72f2d9b228d1768db936fc76dc6ef196/',
            '7fd19bf12a7bcbb627abea8b874bd136/',
            '8757444d0e17c59775b273562693e60b/',
            'a5a4a16f90de7446962ee306f7bbad71/',
            'a617f3d466a4cbb8b388c1f11e585120/',
            'be94e972bd57e32744683444d20a89ef/',
            'd0d2a96b261eccdda3ca7e6f18c7c7e6/',
            'e4d05841e17a92895b06468d09ac73e9/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = '7767efce8e9de6921996ce61db427e54/'
elif simulation == 'lower_etoi':
    from compute_scaling_experiments_helpers import state_lower_e_to_i as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_lower_e_to_i/'
    save_data_dir = 'data/scaling_experiment_lower_e_to_i'
    interesting_paths = [
            '12742e24273ad73c879fc396b575e2e9/',
            '2073b9923d7eb9b128c780a5d8ea9ba8/',
            # '2adb41b5a3195668be54a70516897f74/',
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
elif simulation == 'normal_longer_time':
    from compute_scaling_experiments_helpers import state_scaling_longer_time as state
    base_path = '/p/scratch/cjinb33/shimoura1/huvi_fraction_EI/'
    save_data_dir = 'data/scaling_experiment_longer_time'
    interesting_paths = [
            '028fe4ee6cb64e07220df0d9fa52bfcc/',
            '09e43f2aaa43abb4fca80d2c83c45009/',
            '1637dcc755d2aab4fcb77ed000be5108/',
            '170aec8fbe72cffabfb9dfad48f1cb5e/',
            '227b4854e305a8927999d82e32b8d83b/',
            '34876107e16c952b639b4b115120d161/',
            '3f2b943ce302c7638873b533ca30e884/',
            '5d40feb84afec81f2116b1fd590b0837/',
            '6c0ff4c3e733cf0fd1dc9485545cf027/',
            '6cefaea97aa6fecdce26a56ef6f6be00/',
            '72f2d9b228d1768db936fc76dc6ef196/',
            '7fd19bf12a7bcbb627abea8b874bd136/',
            '8757444d0e17c59775b273562693e60b/',
            # 'a5a4a16f90de7446962ee306f7bbad71/', #6.9
            'a617f3d466a4cbb8b388c1f11e585120/',
            # 'be94e972bd57e32744683444d20a89ef/', #7.0
            'd0d2a96b261eccdda3ca7e6f18c7c7e6/',
            # 'e4d05841e17a92895b06468d09ac73e9/', #7.5
            ]
    sim_path = '01692e3bc76da490797b218b52f4f4da/'
    ana_path = '42d5c4e7288419e0548b161d43b60372/'
elif simulation == 'distributed_params':
    from compute_scaling_experiments_helpers import state_scaling_distributed_params as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_distributed_parameters/'
    save_data_dir = 'data/scaling_experiment_distributed_parameters'
    interesting_paths = [
            '12096673596d51e7d8684e3e0a364151/',
            '22d0bb12e1feeb6006e0954623fe1c3e/',
            '2bd5c38204e617649ccbb6f2b9cb89b3/',
            '361b238d3de0ce8f6608999f1b339e42/',
            '373d8aa4fa4553bc6b8a0916f5f360ca/',
            '39c52ba68922b1aaa56e0582e928469b/',
            '39d1cd29dcba7ce3e91d30df3b3c4230/',
            '3ac091bd6083e24725df3d84aa38b976/',
            '76a2fe05a24c63d7fa7782cf2915bc03/',
            '7de40a139306b6550fd10447615cda3c/',
            '8175030ea48c1915013c1a268734f25f/',
            '82a6d2864abfad496b0e93be437dff9f/',
            '8ecac2dd506b67a91bc059ad8f1b547c/',
            '9e1ac2fa328ac949a6c9d36d771d94d6/',
            'b5ba8abc628cb797b7a67e30e94ebf56/',
            'f9a23fa0df092d966b99b6345a83a893/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = '7767efce8e9de6921996ce61db427e54/'
elif simulation == 'different_synaptic_time_constants':
    from compute_scaling_experiments_helpers import state_scaling_different_synaptic_time_constants as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_different_synaptic_time_constants/'
    save_data_dir = 'data/scaling_experiment_different_syn_time_consts'
    interesting_paths = [
            '12a4da8a81db0bb0104538ab62e70787/',
            '19cbc7395138cf4dd6d1985f0f017a76/',
            '28230684341b372703d5fd053599b053/',
            '2dad5e394de8d46a44598dd8fed9f0b3/',
            '31304ca253f09fd225f0329e685a566f/',
            '33898d285e242f4d6ee29949dc51b6d0/',
            '424a2b5b8564741edbb34c5d2d35cf84/',
            '74b5990035ca965a4cfb56d2b709d389/',
            '8064d110c4f233525f24456043103a27/',
            'a3c4b7a6b7989ef6190cf2e6fbaf8ffe/',
            'c285a602cd9d3e00237062908021e791/',
            'da37dbbadb3952c6ff729dc9b398e327/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = '7767efce8e9de6921996ce61db427e54/'
elif simulation == 'smaller_g':
    from compute_scaling_experiments_helpers import state_scaling_smaller_g as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_smaller_g/'
    save_data_dir = 'data/scaling_experiment_smaller_g'
    interesting_paths = [
            '31b0892554dc7f132d2f4b57e5ab0ab9/',
            # '46d7b79943071ee85ad9b51860419aab/',
            '532c1843dd4be91660b93d8687c77344/',
            '6f9e8c9d30c05546e97889b586d4eb03/',
            '71b25eec7c5dfc3a05806f2c99b25e51/',
            '80939344cb2100978d382e13a33b1afc/',
            # '9ec618e7466c1a26dca7147710f2520f/',
            'b350bd53760fcd1d341ae23e8f5a6de6/',
            # 'c301e0e56aa73e3a25e3f4f915706073/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = '7767efce8e9de6921996ce61db427e54/'
elif simulation == 'lichtman':
    from compute_scaling_experiments_helpers import state_scaling_lichtman as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_lichtman/'
    save_data_dir = 'data/scaling_experiment_lichtman'
    interesting_paths = [
            '0b6c2c62c1ca9095fe14ea70b071a38a/',
            '165aedfb2e860706956af7bae8327b7c/',
            '3ed7438024a2e8d0b875f350284c167b/',
            '40ba75ef3a160ba6a6d782204d61a914/',
            '4be77d9f51eceedfb16fd613815cca92/',
            '54a09749a6415623263a86db75633140/',
            '686e95620c4c93612b32412c152d3313/',
            '902488836de39263d19403e9772c586c/',
            '95b1ad582139d578a2bdf51d07784ee8/',
            'b7b76c9444f081f5a33f5344f9e8bb48/',
            'bb0b17d4e8310d8d489a318e5d0debb7/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = '7767efce8e9de6921996ce61db427e54/'
elif simulation == 'lichtman_chii2':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2 as state
    base_path = '/p/scratch/cjinb33/shimoura1/huvi_fraction_EI_stable_localEtoI1_lichtman_chiI2/'
    save_data_dir = 'data/scaling_experiment_lichtman_chii2'
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
            '95dbe4aba2b7b6fcfa8b215ad4b9ff36/',
            '9f692a9d98028fb6ef1fec9f56c42acb/',
            'a413e5bbaa628c51b03261c4e8c7ef1b/',
            'a8586e55554e8bb52a6b2bfd2cdb5423/',
            'b5816c8150c88676484652da56d5d603/',
            'b6ecf4c0943f997a0f270b90d863cced/',
            'f0a248e8f586b22fc3c00fa7ec02720b/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = 'd8057e7a966caf3eefdc9d234a0785d3/'
elif simulation == 'lichtman_chii2_smaller_g':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_smaller_g as state
    base_path = '/p/scratch/cjinb33/shimoura1/huvi_fraction_EI_stable_localEtoI1_lichtman_chiI2_smaller_g/'
    save_data_dir = 'data/scaling_experiment_lichtman_chii2_smaller_g'
    interesting_paths = [
            '08e8618517e6e057db688d6257680da6/',
            # '2221ff96be64add80bd566545f33c256/',
            '35022b8c38368b83fb671cfc1087d83f/',
            '59cc1c6418e57a0cac22d007f9e0467b/',
            # '5da848b358849f2c2cbdc392eccab7ab/',
            # '6211ac11a7533c167f387f56f19b7dc6/',
            '7d86a0be95d45469743dffa2b2809b8a/',
            # '9e5c047d0d4fda3ad68fff3116746a01/',
            # 'c1b84f0ed08f7e62e539c53c3c1b30c0/',
            'dbbc477edceb3323b0883c89f8d7d955/',
            ]
    sim_path = '1245642e017773b56bc6ea5d9abfd004/'
    ana_path = 'd8057e7a966caf3eefdc9d234a0785d3/'
elif simulation == 'lichtman_chii2_different_time_consts':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_different_time_consts as state
    base_path = '/p/scratch/cjinb33/shimoura1/huvi_fraction_EI_stable_localEtoI1_lichtman_chiI2_different_synaptic_time_constants/'
    save_data_dir = 'data/scaling_experiment_lichtman_chii2_different_time_consts'
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
    save_data_dir = 'data/scaling_experiment_lichtman_chii2_distributed_params'
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
elif simulation == 'lichtman_chii2_different_seed':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_different_seed as state
    base_path = '/p/scratch/cjinb33/jinb3330/huvi_fraction_EI_stable_localEtoI1_lichtman_chiI2/'
    save_data_dir = 'data/scaling_experiment_lichtman_chii2_different_seed'
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
            # '95dbe4aba2b7b6fcfa8b215ad4b9ff36/',
            '9f692a9d98028fb6ef1fec9f56c42acb/',
            'a413e5bbaa628c51b03261c4e8c7ef1b/',
            'a8586e55554e8bb52a6b2bfd2cdb5423/',
            'b5816c8150c88676484652da56d5d603/',
            'b6ecf4c0943f997a0f270b90d863cced/',
            'f0a248e8f586b22fc3c00fa7ec02720b/',
            ]
    sim_path = 'de4934b8c7777751f7c516e2ad35f50a/'
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
