import os
from os.path import join as path_join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_firing_rate_distribution(filepath, figname):
    """
    Plot firing rate distribution for each population across different areas.

    Parameters
    ----------
    filepath : str
        Path to the directory containing the pickle files.
    figname : str
        Name of the output figure.
    """

    # Read the rates_individual pickle file
    rates_individual = pd.read_pickle(path_join(filepath, 'rates_individual.pkl'))

    # Plot 8 figures in a 4x2 grid the rates of each population across areas
    fig, ax = plt.subplots(4, 2, figsize=(5.63, 6), sharex=True)  # Increase figure size for readability

    row, col = 0, 0  # Initialize subplot counters
    layer_labels = ['2/3', '4', '5', '6']

    for layer in rates_individual.index.get_level_values(1).unique():
        for pop in rates_individual.index.get_level_values(2).unique():
            # Initialize list to store firing rate distributions
            rates_dist = []

            # Calculate firing rate distribution for each area
            for area in rates_individual.index.get_level_values(0).unique():
                if (layer, pop) not in rates_individual[area]:
                    print('Skipping ' + area + ' ' + layer + ' ' + pop)
                    continue
                else:
                    vals, bins = np.histogram(rates_individual[area, layer, pop], bins=np.arange(.1, 100., .5))
                    if np.sum(vals) > 0:
                        vals = vals / np.sum(vals)
                    rates_dist.append(vals)

            # Calculate mean and standard deviation of firing rate distributions
            mean_rates = np.mean(rates_dist, axis=0)
            std_rates = np.std(rates_dist, axis=0)

            # Plot firing rate distribution
            ax[row, col].plot(bins[:-1], mean_rates, linewidth=2, color='black')
            ax[row, col].fill_between(bins[:-1], mean_rates-std_rates, mean_rates+std_rates, alpha=0.5, color='black')
            ax[row, col].set_title(layer_labels[row] + ' ' + pop)
            ax[row, col].set_xscale('log')
            ax[row, col].set_xlim(0.1, 100)

            col += 1
            if col >= 2:
                col = 0
                row += 1

    # Add overall figure title
    fig.supylabel('Density')
    fig.supxlabel('Firing rate (spikes/s)')

    fig.tight_layout()
    fig.savefig(f'figures/{figname}.pdf')


# Set the filepath and figname
filepath = os.path.join(os.getcwd(), 'out/8c49a09f51f44fbb036531ce0719b5ba/4772f0b020c9f3310f4096a6db758343/')
figname = "figure_firing_rate_distribution_bestfit"

# Call the plot_firing_rate_distribution function
plot_firing_rate_distribution(filepath, figname)
