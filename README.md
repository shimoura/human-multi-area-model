# Multi-Scale Spiking Network Model of Human Cerebral Cortex

This code implements the multi-scale, spiking network model of human cortex developed at the Institute of Advanced Simulation (IAS-6), Research Center Jülich. The model has been documented in the following publication:

- [1] Pronold, J., Meegen, A. van, Shimoura, R. O., Vollenbröker, H., Senden, M., Hilgetag, C. C., Bakker, R., & Albada, S. J. (2024). Multi-scale spiking network model of human cerebral cortex. Cerebral Cortex. [https://doi.org/10.1093/cercor/bhae409](https://doi.org/10.1093/cercor/bhae409).

![model_overview](./figures/model_overview.png)

**Model overview**: The model comprises all 34 areas of the Desikan-Killiany parcellation in one hemisphere of human cerebral cortex. Each area is modeled by a column with $\mathrm{1\mathrm{mm^{2}}}$ cortical surface. Within each column, the full number of neurons and synapses based on anatomical data is included. In total, this leads to 3.47 million neurons and 42.8 billion synapses. Both the intrinsic and the cortico-cortical connectivity are layer- and population-specific.

## Table of contents
- [Multi-Scale Spiking Network Model of Human Cerebral Cortex](#multi-scale-spiking-network-model-of-human-cerebral-cortex)
  - [Table of contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
    - [Data](#data)
    - [Requirements](#requirements)
  - [Installation](#installation)
    - [Python modules using Anaconda](#python-modules-using-anaconda)
    - [NEST installation](#nest-installation)
  - [Code repository](#code-repository)
  - [How to run](#how-to-run)
    - [Configuration](#configuration)
    - [Run on a cluster](#run-on-a-cluster)
    - [Ploting figures](#ploting-figures)
  - [Collaborators](#collaborators)
  - [Acknowledgments](#acknowledgments)
  - [How to cite](#how-to-cite)

## Prerequisites
### Data

Data extracted from experimental references and necessary to run the codes can be found in [experimental_data/](./experimental_data/) folder. These files will be automatically loaded when running the simulation (check section [How to run](#how-to-run) for details).

Please note that some data has to be manually downloaded. Specifically, the data stored in `./experimental_data/macaque/` and `./experimental_data/rutishauser/`. Both data are only required for specific plots, not being necessary to run the examples presented here.

### Requirements

The entire workflow of the model, from data preprocessing through the simulation to the final analysis, relies on the `Python` programming language. The complete list of Python packages with the specific version we used to run our simulations can be found in ```humam.yml``` file. Other package versions may not work properly, specifically, using `Pandas >= 1.0` will raise an error when creating the network.

All network simulations were performed using the `NEST simulator` version `2.20.2` (https://www.nest-simulator.org/).

## Installation

### Python modules using Anaconda
The Python modules can be installed with the [Anaconda](https://www.anaconda.com/download) data science platform or via its free minimal installer called [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html) (recommended). 

Most dependencies are handled using ```conda```. On a cluster, ```snakemake``` automatically creates the conda environment for you. On a local computer, simply run:
```
conda env create -f humam.yml
```  
This command will create a ```conda``` environment and automatically install all Python packages defined in the ```humam.yml``` file. 
**Note**: currently the model is not adapted to run on a local computer because of the memory requirements. A downscaling factor option will be implemented in the future. 

The ```NEST simulator``` is not included in this file, although it can be installed via ```conda``` too. We opt to keep an independent installation of NEST so we can better control the version being used.

### NEST installation

All different methods to install NEST can be found in their extensive documentation: https://nest-simulator.readthedocs.io/en/stable/installation/index.html. 
In our published results, we used the "Install NEST from source" option with NEST ```2.20.2``` version. Note: after the step to clone nest-simulator from Github, change the branch with: ```git checkout v2.20.2```.

The NEST installation path will have to be specified in `config.yaml` described in the ["How to run"](#how-to-run) section.

## Code repository

Folder structure:
| directory | description |
| --- | --- |
| [./experimental_data/](./experimental_data/) | contains experimental datasets used for building the network and for comparing results |
| [./experiments/](./experiments/) | contains python scripts which set the model parameters for different simulation experiments |
| [./figures/](./figures/) | output directory for figures |
| [./misc/](./misc/) | includes supplementary files such as code documentation ([/docs](./misc/docs/)), matplotlib style files ([/mplstyles](./misc/mplstyles/)), and other experiment files ([/experiments](./misc/experiments/))
| [./out/](./out/) | directory where the simulation output is stored |
| [./src/](./src/) | main directory with python scripts to run the network simulation |
| [./simulated_data/](./simulated_data/) | simulated data generated from scaling experiments |

Brief description of the main files in [./src/](./src/) directory:

| script | description |
| --- | --- |
| `./network.py` | python class that gathers and prepares all data for setting up the NEST simulation |
| `./simulation.py` | python class that setups and builds the network for running the simulations |
| `./analysis.py` | python class that provides functions to analyse simulation results |
| `./default_` | scripts that define the default network, simulation and analysis parameter dictionaries |
| `./snakemake_` | helper scripts which use an `experiment.py` file to create, simulate, and analyse the network |
| `./figure_` | scripts that plot specific figures showed in our publication [1] |
| `./compute_` | scripts to compute the scalling experiment |
  
Additionally, in [./src/](./src/) directory you can also find the following subfolders:
| directory | description |
| --- | --- |
| [./src/data_loader/](./src/data_loader/) | contains auxiliary scripts for loading the data used for building the network |
| [./src/data_preprocessing/](./src/data_preprocessing/) | contains auxiliary scripts for preprocessing the data used for building the network |
| [./src/helpers](./src/helpers/) | contains auxiliary helper scripts |
| [./src/theory/](./src/theory/) | contains the scripts used for the mean-field analysis |

## How to run

The example below shows how to prepare the configuration files and how to run the code. 
All the workflow is managed using the [Snakemake](https://snakemake.readthedocs.io/en/stable/#) tool. To run different network setups or experiments, the user has only to set the parameters in a Python script (two examples are shown in [./experiments/](./experiments/)) and simulate following the instructions below.

### Configuration

Create a `config.yaml` file inside the repository's main directory. An example is shown in `config_jureca.yaml`. Please note that the NEST path should be given as: `<path_to_NEST_installation>/install/`. 
If running in a cluster, you also have to define the cluster configurations on `cluster.json` file. An example is given as well, but you should modify it accordingly with your cluster configuration.

**NOTE**: the current version of the code has no downscaling factor to run a smaller version of the network, which limits its usage on a local computer. 
This will be implemented in a future version.

### Run on a cluster

To run the model on a cluster, ensure you have a working `conda` and `snakemake` installation. 


Start with
```
conda activate humam
```
to add `conda` to the `PATH`. Lastly start `snakemake` with the cluster specification:
```
bash snakemake_slurm.sh
```

**NOTE**: to run the current version on JURECA cluster (Jülich Supercomputing Centre at Forschungszentrum Jülich), it is recommended to use the modules defined in `config_jureca.yaml` file instead of the conda environment. If so, remove the `--use-conda` flag in the `snakemake_slurm.sh` script before running the code line above.

This script will run the workflow defined in `Snakefile`, which follows the sequence:
1. read all `*.py` experiment files contained in the `./experiments/` directory. **NOTE**: If you want to run fewer/more experiments, remove/add these files from the `./experiments/` directory.
2. load necessary modules for MPI and NEST before executing
3. create the network
4. simulate the network
5. analyse the results from simulation

By default, the resulting simulation data will be stored in the `./out` directory. For each experiment, different information regarding network, simulation, and analysis details will be stored following this structure:

`./out/<network_hash>/<simulation_hash>/<analysis_hash>`

The hash identifiers are automatically generated based on the set of parameters defined on the network, simulation, and analysis dictionaries.

### Ploting figures

After running the complete workflow described in ["Run on a cluster"](###run-on-a-cluster) section, you can find the raster plots and other figures automatically generated from the simulation results in

`./out/<network_hash>/<simulation_hash>/<analysis_hash>/plots/` 

Other figures shown in [1] can be manually plotted using the scripts in `./src/` named as "figure_*". These figures are stored at `./figures/`. For instance, after running the ground and best-fit state experiments, from the main directory you can plot figures 4 and 6 presented in [1] by running the script:

```
python src/figure_spike_statistics.py
```

## Collaborators

The scientific content contributions were made by the authors of the publication [1]: Jari Pronold, Alexander van Meegen, Renan O. Shimoura, Hannah Vollenbröker, Mario Senden, Claus C. Hilgetag, Rembrandt Bakker, and Sacha J. van Albada.

## Acknowledgments
We thank Sebastian Bludau and Timo Dickscheid for helpful discussions about cytoarchitecture and parcellations. Furthermore, we gratefully acknowledge all the shared experimental data that underlies our work, and the effort spent to collect it.

This work was supported by the German Research Foundation (DFG) Priority Program “Computational Connectomics” (SPP 2041; Project 347572269), the European Union’s Horizon 2020 Framework Programme for Research and Innovation under Specific Grant Agreement No. 945539 (Human Brain Project SGA3), the European Union’s Horizon Europe Programme under the Specific Grant Agreement No. 101147319 (EBRAINS 2.0 Project), the Joint Lab “Supercomputing and Modeling for the Human Brain”, and HiRSE_PS, the Helmholtz Platform for Research Software Engineering - Preparatory Study, an innovation pool project of the Helmholtz Association. The use of the JURECA-DC supercomputer in Jülich was made possible through VSR computation time grant JINB33 (“Brain-Scale Simulations”). Open access publication funded by the German Research Foundation (DFG), project 491111487.

## How to cite
If you use this code, please cite:
- Pronold, J., Meegen, A. van, Shimoura, R. O., Vollenbröker, H., Senden, M., Hilgetag, C. C., Bakker, R., & Albada, S. J. (2024). Multi-scale spiking network model of human cerebral cortex. Cerebral Cortex. [https://doi.org/10.1093/cercor/bhae409](https://doi.org/10.1093/cercor/bhae409).
