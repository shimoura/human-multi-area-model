# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.io import loadmat


class VonEconomoKoskinas():
    """
    Loads the structural data of von Economo & Koskinas.
    Provided by Alex Goulas and Claus Hilgetag.


    Parameters
    ----------
    file_path : string
        Path to data file
    """

    def __init__(self, file_path):
        self.file_path = file_path

        # Load data into DataFrame
        self.data = pd.read_excel(
            self.file_path,
            sheet_name='Sheet1',
            index_col=[0, 2]
        )

        # Keep only cell density and layer thickness & rename
        self.data = self.data[[
            u"mean cell content per layer [cells/mm³]",
            u"Layer thickness overall [mm]"
        ]]
        self.data = self.data.rename(columns={
            u"mean cell content per layer [cells/mm³]": "neuron_density",
            u"Layer thickness overall [mm]": "layer_thickness"
        })

        # Cast and replace not castable
        self.data.where(self.data != '-', 0, inplace=True)
        self.data.where(self.data != '60.000 (?)', 60000, inplace=True)
        self.data['layer_thickness'] = self.data['layer_thickness'].astype(
            np.float64)
        self.data['neuron_density'] = self.data['neuron_density'].astype(
            np.float64)

        # Fill nas due to cell merge in layer thickness by 0
        self.data["layer_thickness"] = self.data["layer_thickness"].fillna(
            value=0
        )
        # Fill nas due to cell merge in neuron density by previous value
        self.data["neuron_density"] = self.data["neuron_density"].fillna(
            value=0
        )

    def __repr__(self):
        return ('Von Economo & Koskinas structural data\n'
                'File: {0}\nAreas (index lvl 0):\n{1}\n'
                'Layers (index lvl 1):\n{2}\nColumns:\n{3}\n'
                .format(self.file_path,
                        self.data.index.levels[0].values,
                        self.data.index.levels[1].values,
                        self.data.columns.values)
                )

    def getNeuronDensity(self):
        """
        Returns the neuron density in cells/mm^3.

        Returns
        -------
        neuron_density : Series
        """
        return self.data['neuron_density']

    def getLayerthickness(self):
        """
        Returns the layer thickness in mm.

        Returns
        -------
        layer_thickness : Series
        """
        return self.data['layer_thickness']

    def getAreaList(self):
        """
        Returns the list of areas.

        Returns
        -------
        area_list : list
        """
        return self.data.index.levels[0].values

    def dump(self, file_path, key=''):
        """
        Save all data as a json file
        """
        # TODO keep as hdf?
        self.data.to_hdf(file_path, key='von Economo Koskinas', mode='w')


class HcpDesikanKilliany():
    """
    Connectivity from the HCP in the Desikan Killiany atlas.
    Provided by Alex Goulas and Claus Hilgetag.

    Returns
    -------
    data : Panel
        All available data
    names : list
        Names of Desikan Killiany atlas
    """

    def __init__(self, file_path):
        self.file_path = file_path

        # Read in data
        self.data = loadmat(self.file_path)

        # Preprocess names: Convert to 1d list & strip leading 'ctx-lh-'
        self.names = self.data['DK_Names']
        self.names = np.concatenate(self.names.flatten()).tolist()
        self.names = [s.split('-')[-1] for s in self.names]

        self.dist = pd.DataFrame(
                self.data['Dist'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

        self.RightHem = pd.DataFrame(
            self.data['LowResDK_EstimatedDist_RHem'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

        self.LeftHem = pd.DataFrame(
            self.data['LowResDK_EstimatedDist_LHem'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

        self.ConnectivityRight = pd.DataFrame(
            self.data['C_R'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

        self.ConnectivityLeft = pd.DataFrame(
            self.data['C_L'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

    def __repr__(self):
        return ('HCP connectivity data\n'
                'File: {0}\nRegions (major & minor axis):\n{1}\n'
                'Items:\n{2}\n'
                .format(self.file_path,
                        np.array(self.names),
                        self.data.items.values)
                )

    def getDesikanKillianyNames(self):
        """
        Returns the names of the different Desikan Killiany areasself.

        Returns
        -------
        names : list
        """
        return self.names

    def getDist(self):
        """
        Returns the euclidean distances in mm of the different Desikan Killiany
        areas.

        Returns
        -------
        euclidean_distance : DataFrame
        """
        return self.dist

    def getFiberLengthRight(self):
        """
        Returns the fiber lengthes in mm of the right hemisphere of the different
        Desikan Killiany areas.

        Returns
        -------
        fiber_length_right : DataFrame
        """
        return self.RightHem

    def getFiberLengthLeft(self):
        """
        Returns the fiber lengthes in mm of the left hemisphere of the different
        Desikan Killiany areas.

        Returns
        -------
        fiber_length_left : DataFrame
        """
        return self.LeftHem

    def getConnectivityRight(self):
        """
        Returns the connectivity of the right hemisphere of the different
        Desikan Killiany areas.

        Returns
        -------
        connectivity_right : DataFrame
        """
        return self.ConnectivityRight

    def getConnectivityLeft(self):
        """
        Returns the connectivity of the left hemisphere of the different
        Desikan Killiany areas

        Returns
        -------
        connectivity_left : DataFrame
        """
        return self.ConnectivityLeft

    def dump(self, file_path):
        """
        Save all data as a hdf file
        """
        # TODO keep as hdf?
        self.data.to_hdf(file_path, key='HCP', mode='w')


class VolumesDK():
    """
    Loads area volumes in Desikan Killiany parcellation. Provided
    by Alex Goulas.

    Parameters
    ----------
    file_path : string
        Path to data file
    """

    def __init__(self, file_path):
        self.file_path = file_path

        # Read in data
        self.data = loadmat(self.file_path)

        # Preprocess names: Convert to 1d list & strip leading 'ctx-lh-'
        self.names = self.data['HumanDKNames']
        self.names = np.concatenate(self.names.flatten()).tolist()
        self.data = pd.Series(
            data=self.data['Volume'].flatten(),
            index=self.names,
            dtype=np.float64
        )

    def getVolume(self):
        """
        Returns the volume of all areas in the Desikan Killiany
        parcellation. The volume is cubic mm.

        Returns
        -------
        volume : Series
            Volume of the areas
        """
        return self.data


if __name__ == "__main__":

    vek_path = '../data/hilgetag/StructuralData_VonEconomoKoskinas.xls'
    vek = VonEconomoKoskinas(vek_path)
    print(vek)
    print(vek.getNeuronDensity())
    print(vek.getLayerthickness())
    print(vek.getAreaList())
    # vek.dump('vek_tmp.hdf')

    # dk_path = 'data/hilgetag/Connectivity_Distances_HCP_DesikanKilliany.mat'
    # dk = HcpDesikanKilliany(dk_path)
    # print(dk)
    # dk.dump('dk_tmp.hdf')
    # print(dk.data['Connectivity (right) [NOS]', 'middletemporal'])

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(
    #    dk.data['Connectivity (right) [NOS]'],
    #    ax=ax,
    #    xticklabels=1,
    #    yticklabels=1,
    #    cmap='Blues'
    # )
    # plt.tight_layout()
    # plt.show()
