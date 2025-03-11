import pandas as pd


def load_receptor_densities(path):
    """
    Provides the ratio of NMDA to AMPA receptor distribution.

    Returns
    -------
    ratio : Series
        Layer-resolved NMDA to AMPA ratios.
    """
    return pd.read_csv(path, index_col=[0,1,2]).squeeze()

def load_receptor_densities_std(path):
    """
    Provides the ratio of NMDA to AMPA receptor distribution.

    Returns
    -------
    ratio : Series
        Layer-resolved NMDA to AMPA ratios.
    """
    path = path[:-4] + "_std.csv"
    return pd.read_csv(path, index_col=[0,1,2]).squeeze()