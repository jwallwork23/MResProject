import numpy as np

# Change backend to resolve framework problems:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def gauge_timeseries(gauge, dat):
    """
    Store timeseries data for a particular gauge.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param dat: a list of data values of this gauge.
    :return: a file containing the timeseries data.
    """

    name = raw_input('Enter a name for this time series: ')
    outfile = open('timeseries/{y1}_{y2}.txt'.format(y1=gauge, y2=name), 'w+')

    for i in range(len(dat)):
        outfile.write(str(dat[i]) + '\n')

    outfile.close()


def plot_gauges(gauge):
    """
    Plot timeseries data on a single axis.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :return: a matplotlib plot of the corresponding gauge timeseries data.
    """

    setup = {1: 'coarse',           # 3,126 vertices
             2: 'medium'}           # 25,976 vertices
             # 3: 'fine',             # 226,967 vertices
             # 4: 'anisotropic',
             # 5: 'goal-based'}
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Loop over mesh resolutions:
    for key in setup:
        val = []
        infile = open('timeseries/{y1}_{y2}.txt'.format(y1=gauge, y2=setup[key]), 'r')
        for line in infile:
            val.append(float(line))
        infile.close()
        plt.plot(np.linspace(0, 60, len(val)), val)     # Plot time series for this setup

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([-5, 5])
    plt.legend(loc='upper right')
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')
    plt.show()
    plt.savefig('plots/tsunami_outputs/screenshots/full_gauge_timeseries_{y}.png'.format(y=gauge))
