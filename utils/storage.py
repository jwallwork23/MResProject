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

    name = raw_input('Enter a name for this time series (e.g. xcoarse): ')
    outfile = open('timeseries/{y1}_{y2}.txt'.format(y1=gauge, y2=name), 'w+')
    for i in range(len(dat)):
        outfile.write(str(dat[i]) + '\n')
    outfile.close()


def plot_gauges(gauge, problem='comparison'):
    """
    Plot timeseries data on a single axis.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param problem: problem type name string, corresponding to either 'verification' or 'comparison'.
    :return: a matplotlib plot of the corresponding gauge timeseries data.
    """

    if problem == 'comparison':
        setup = {0: 'xcoarse',                      # 3,126 vertices
                 1: 'medium',                       # 25,976 vertices
                 2: 'fine',                         # 97,343 vertices
                 3: 'anisotropic_point85scaled_rm=30',
                 4: 'goal-based'}
        labels = {0: 'Fixed, coarse mesh',
                  1: 'Fixed, intermediate mesh',
                  2: 'Fixed, fine mesh',
                  3: 'Anisotropic adapted mesh',
                  4: 'Goal-based adapted mesh'}
    else:
        setup = {0: 'fine',
                 1: 'fine_rotational',
                 2: 'fine_nonlinear',
                 3: 'fine_nonlinear_rotational'}
        labels = {0: 'Linear, non-rotational equations',
                  1: 'Linear, rotational equations',
                  2: 'Nonlinear, non-rotational equations',
                  3: 'Nonlinear, rotational equations'}
    styles = {0: ':', 1: '--', 2: '-', 3: '-.', 4: ':'}
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    plt.clf()

    # Temporary user specified input for incomplete plots:
    progress = int(raw_input('How far have we got for this gauge? (1/2/3/4/5): ') or 4)

    # Loop over mesh resolutions:
    for key in range(progress):
        val = []
        i = 0
        v0 = 0
        infile = open('timeseries/{y1}_{y2}.txt'.format(y1=gauge, y2=setup[key]), 'r')
        for line in infile:
            if i == 0:
                i += 1
                v0 = float(line)
            val.append(float(line) - v0)
        infile.close()
        if setup[key] in ('fine_nonlinear', 'fine_nonlinear_rotational', 'anisotropic_point85scaled_rm=30'):
            plt.plot(np.linspace(0, 25, len(val)), val, label=labels[key], linestyle=styles[key])
        else:
            plt.plot(np.linspace(0, 60, len(val)), val, label=labels[key], linestyle=styles[key])
    plt.gcf()
    plt.ylim([-5, 5])
    plt.legend(loc=4)       # 'upper right' == 1 and 'lower right' == 4
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')
    plt.savefig('plots/tsunami_outputs/screenshots/full_gauge_timeseries_{y1}_{y2}.png'.format(y1=gauge, y2=problem))
