# Multi-Scale Spiking Network Model of Human Cerebral Cortex

This code implements the multi-scale, spiking network model of human cortex developed at the Institute of Neuroscience and Medicine (INM-6), Research Center Jülich. The model has been documented in the following publication (to be updated once it is published in a peer-reviewed journal):

- [1] Pronold, J., Meegen, A. van, Vollenbröker, H., Shimoura, R. O., Senden, M., Hilgetag, C. C., Bakker, R., & Albada, S. J. van. (2023). Multi-Scale Spiking Network Model of Human Cerebral Cortex. BioRxiv, 2023.03.23.533968. [https://doi.org/10.1101/2023.03.23.533968](https://doi.org/10.1101/2023.03.23.533968)

![model_overview](./figures/model_overview.png)

**Model overview**: The model comprises all 34 areas of the Desikan-Killiany parcellation in one hemisphere of human cerebral cortex. Each area is modeled by a column with $\mathrm{1\:\mathrm{mm^{2}}}$ cortical surface. Within each column, the full number of neurons and synapses based on anatomical data is included. In total, this leads to 3.47 million neurons and 42.8 billion synapses. Both the intrinsic and the cortico-cortical connectivity are layer- and population-specific.

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
  - [Collaborators](#collaborators)
  - [(TODO) Get support](#todo-get-support)
  - [Acknowledgments](#acknowledgments)
  - [How to cite](#how-to-cite)

## Prerequisites
### Data

Data extracted from experimental references and necessary to run the codes can be found in `data/` folder. These files will be automatically loaded when running the simulation (check section [How to run](#how-to-run) for details).

### Requirements

The entire workflow of the model, from data preprocessing through the simulation to the final analysis, relies on the `Python` programming language version `3.9` or above. The complete list of Python packages we used to run our simulations can be found in ```huvi.yml``` file. 

All network simulations were performed using the `NEST simulator` version `2.20.2` (https://www.nest-simulator.org/).

**(TODO)** Hardware requirements:
- ?? MB memory
- ...

## Installation

### Python modules using Anaconda
The Python modules can be installed with the [Anaconda](https://www.anaconda.com/download) data science platform or via its free minimal installer called [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html) (recommended). 

Most dependencies are handled using ```conda```. On a cluster, ```snakemake``` automatically creates the conda environment for you. On a local computer, simply run:
```
conda env create -f huvi.yml
```  
This command will create a ```conda``` environment and automatically install all Python packages defined in the ```huvi.yml``` file. 
**Note**: currently the model is not adapted to run on a local computer because of the memory requirements. A downscaling factor option will be implemented in the future. 

The ```NEST simulator``` is not include in this file, although it can be installed via ```conda``` too. We opt to keep an independent installation of NEST so we can better control the version being used.

### NEST installation

All different methods to install NEST can be found in their extensive documentation: https://nest-simulator.readthedocs.io/en/stable/installation/index.html. 
In our published results, we used the "Install NEST from source" option with NEST ```2.20.2``` version. Note: after the step to clone nest-simulator from Github, change the branch with: ```git checkout v2.20.2```.

The NEST installation path will have to be specified in `config.yaml` described in the ["How to run"](#how-to-run) session.

## Code repository

Folder structure:
- `./data`: contains experimental data-sets used for building the network and for comparing results
- `./docs`: **TODO** 
- `./experiments`: contains python scripts which set the model parameters for different simulation experiments
- `./figures`: **TODO**
- `./mplstyles`: **TODO**
- `./out`: directory where the simulation output is stored
- `./python`: main directory with python scripts to run the network simulation

Brief description of the main files in `./python` directory:
- `network.py`: python class that gathers and prepares all data for setting up the NEST simulation
- `simulation.py`: python class that setups and builds the network for running the simulations
- `analysis.py`: python class that provides functions to analyse simulation results
- `default_`: scripts that define the default network, simulation and analysis parameter dictionaries
- `snakemake_`: helper scripts which use an `experiment.py` file to create, simulate, and analyse the network
- `figure_`: scripts that plot specific figures showed in our publication [1]
- `compute_`: scripts to compute the scalling experiment
  
Additionally, in `./python/` directory you can also find the following subfolders:
- `./python/data_loader`: contains auxiliary scripts for loading the data used for building the network
- `./python/data_preprocessing`: contains auxiliary scripts for preprocessing the data used for building the network
- `./python/helpers`: **TODO**
- `./python/theory`: **TODO**

## How to run

### Configuration

Create a `config.yaml` file inside the repository's main directory. A minimal example is `config_pc.yaml`, a more involved on `config_blaustein.yaml`.
If running in a cluster, you also have to define the cluster configurations on `cluster.json` file.

### Run on a cluster

To run the model on a cluster, make sure you have a working `conda` and `snakemake` installation on the cluster. 

Start with
```
conda activate base
```
to add `conda` to the `PATH`. Lastly start `snakemake` with the cluster specification:
```
bash snakemake_slurm.sh
```
This script will run the workflow defined in `Snakefile`, which follows the sequence:
1. read all `*.py` experiment files contained in the `./experiments/` directory.
2. load necessary modules for MPI and NEST before executing
3. create the network
4. simulate the network
5. analyse the results from simulation

## Collaborators

The scientific content contributions were made by the authors of the publication [1]: Jari Pronold, Alexander van Meegen, Hannah Vollenbröker, Renan O. Shimoura, Mario Senden, Claus C. Hilgetag, Rembrandt Bakker, and Sacha J. van Albada.

## (TODO) Get support
Describe how to get in touch in case of questions or issues.

Contact
example-mailing-list@support.org

## Acknowledgments
We thank Sebastian Bludau and Timo Dickscheid for helpful discussions about cytoarchitecture and parcellations. Furthermore, we gratefully acknowledge all the shared experimental data that underlies our work, and the effort spent to collect it.

This work was supported by the German Research Foundation (DFG) Priority Program “Computational Connectomics” (SPP 2041; Project 347572269), the European Union’s Horizon 2020 Framework Programme for Research and Innovation under Specific Grant Agreement No. 945539 (Human Brain Project SGA3), the Joint Lab “Supercomputing and Modeling for the Human Brain”, and HiRSE_PS, the Helmholtz Platform for Research Software Engineering - Preparatory Study, an innovation pool project of the Helmholtz Association. The use of the JURECA-DC supercomputer in Jülich was made possible through VSR computation time grant JINB33 (“Brain-Scale Simulations”)

## How to cite
If you use this code, cite the paper publication:
- Pronold, J., Meegen, A. van, Vollenbröker, H., Shimoura, R. O., Senden, M., Hilgetag, C. C., Bakker, R., & Albada, S. J. van. (2023). Multi-Scale Spiking Network Model of Human Cerebral Cortex. BioRxiv, 2023.03.23.533968. [https://doi.org/10.1101/2023.03.23.533968](https://doi.org/10.1101/2023.03.23.533968)