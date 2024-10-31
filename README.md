# Multi-Scale Spiking Network Model of Human Cerebral Cortex

This code implements the multi-scale, spiking network model of human cortex developed at the Institute of Advanced Simulation (IAS-6), Research Center Jülich. The model has been documented in the following publication:

- [1] Pronold, J., Meegen, A. van, Shimoura, R. O., Vollenbröker, H., Senden, M., Hilgetag, C. C., Bakker, R., & Albada, S. J. (2024). Multi-scale spiking network model of human cerebral cortex. Cerebral Cortex. [https://doi.org/10.1093/cercor/bhae409](https://doi.org/10.1093/cercor/bhae409).

![model_overview](./figures/model_overview.png)

**Model overview**: The model comprises all 34 areas of the Desikan-Killiany parcellation in one hemisphere of human cerebral cortex. Each area is modeled by a column with $\mathrm{1\mathrm{mm^{2}}}$ cortical surface. Within each column, the full number of neurons and synapses based on anatomical data is included. In total, this leads to 3.47 million neurons and 42.8 billion synapses. Both the intrinsic and the cortico-cortical connectivity are layer- and population-specific.

## Table of contents
- [Multi-Scale Spiking Network Model of Human Cerebral Cortex](#multi-scale-spiking-network-model-of-human-cerebral-cortex)
  - [Table of contents](#table-of-contents)
  - [Try it on EBRAINS](#try-it-on-ebrains)
    - [User instructions](#user-instructions)
      - [Try it on EBRAINS](#try-it-on-ebrains-1)
      - [Fork the repository and save your changes](#fork-the-repository-and-save-your-changes)
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

## Try it on EBRAINS

Do you want to start using or simply run the model? Click the button below.
**Please note**: make sure you check and follow our User instructions, especially if you plan to make and save the changes, or if you need step-by-step instructions.
<a href="https://lab.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fshimoura%2Fhuman-multi-area-model.git&urlpath=lab%2Ftree%2Fhuman-multi-area-model.git%2Fhumam_tutorial.ipynb&branch=add-downscaling-factor"> <img src="https://nest-simulator.org/TryItOnEBRAINS.png" alt="Try it on EBRAINS" width="260"/></a>

--------------------------------------------------------------------------------

### User instructions
The Jupyter Notebook `humam_tutorial.ipynb` illustrates the simulation workflow with a down-scaled version of the multi-area model. This notebook can be explored and executed online in the Jupyter Lab provided by EBRAINS without the need to install any software yourself.<br>
* Prerequisites: an [EBRAINS](https://www.ebrains.eu/) account. If you don’t have it yet, register at [register page](https://iam.ebrains.eu/auth/realms/hbp/protocol/openid-connect/registrations?response_type=code&client_id=xwiki&redirect_uri=https://wiki.ebrains.eu). Please note: registering an EBRAINS account requires an institutional email.<br>
* If you plan to only run the model, instead of making and saving changes you made, go to [Try it on EBRAINS](#try-it-on-ebrains-1); Should you want to adjust the parameters, save the changes you made, go to [Fork the repository and save your changes](#fork-the-repository-and-save-your-changes).

#### Try it on EBRAINS
1. Click [Try it on EBRAINS](https://lab.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fshimoura%2Fhuman-multi-area-model.git&urlpath=lab%2Ftree%2Fhuman-multi-area-model.git%2Fhumam_tutorial.ipynb&branch=add-downscaling-factor). If any error or unexpected happens during the following process, please close the browser tab and restart the [User instruction](https://lab.ebrains.eu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fshimoura%2Fhuman-multi-area-model.git&urlpath=lab%2Ftree%2Fhuman-multi-area-model.git%2Fhumam_tutorial.ipynb&branch=add-downscaling-factor) process again.
2. On the `Lab Execution Site` page, select a computing center from the given list.
3. If you’re using EBRAINS for the first time, click `Sign in with GenericOAuth2` to sign in on EBRAINS. To do this, you need an EBRAINS account.
4. Once signed in, on the `Server Options` page, choose `Official EBRAINS Docker image 23.06 for Collaboratory.Lab (recommended)`, and click `start`.
5. Once succeeded, you’re now at a Jupyter Notebook named `humam_tutorial.ipynb`. Click the field that displays `Python 3 (ipykernel)` in the upper right corner and switch the `kernel` to `EBRAINS-23.09`.
6. Congratulations! Now you can run the model. Enjoy!<br> To run the model, click the `Run` on the title bar and choose `Run All Cells`. It takes several minutes until you get all results.<br>
**Please note**: every time you click the `Try it on EBRAINS` button, the repository is loaded into your home directory on EBRAINS Lab and it overrides your old repository with the same name. Therefore, make sure you follow the [Fork the repository and save your changes](#fork-the-repository-and-save-your-changes) if you make changes and want to save them.
 
#### Fork the repository and save your changes
With limited resources, EBRAINS Lab regularly deletes and cleans data loaded on the server. This means the repository on the EBRAINS Lab will be periodically deleted. To save changes you made, make sure you fork the repository to your own GitHub, then clone it to the EBRAINS Lab, and do git commits and push changes.
1. Go to our [Human Multi-Area Model](https://github.com/INM-6/human-multi-area-model), create a fork by clicking the `Fork`. In the `Owner` field, choose your username and click `Create fork`. Copy the address of your fork by clicking on `Code`, `HTTPS`, and then the copy icon.
2. Go to [EBRAINS Lab](https://lab.de.ebrains.eu), log in, and select a computing center from the given list.
3. In the Jupyter Lab, click on the `Git` icon on the left toolbar, click `Clone a Repository` and paste the address of your fork.
4. Now your forked repository of human multi-area model is loaded on the server. Enter the folder `human-multi-area-model` and open the notebook `humam_tutorial.ipynb.ipynb`.
5. Click the field that displays `Python 3 (ipykernel)` in the upper right corner and switch the `kernel` to `EBRAINS-23.09`.
6. Run the notebook! To run the model, click the `Run` on the title bar and choose `Run All Cells`. It takes several minutes until you get all results. 
7. You can modify the exposed parameters before running the model. If you want to save the changes you made, press `Control+S` on the keyboard, click the `Git` icon on the most left toolbar, do git commits and push.<br> 
To commit, on `Changed` bar, click the `+` icon, fill in a comment in the `Summary (Control+Enter to commit)` at lower left corner and click `COMMIT`.<br> 
To push, click the `Push committed changes` icon at upper left which looks like cloud, you may be asked to enter your username and password (user name is your GitHUb username, password should be [Personal access tokens](https://github.com/settings/tokens) you generated on your GitHub account, make sure you select the `repo` option when you generate the token), enter them and click `Ok`.
1. If you would like to contribute to our model or bring your ideas to us, you’re most welcome to contact us. It’s currently not possible to directly make changes to the original repository, since it is connected to our publications.

--------------------------------------------------------------------------------

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
