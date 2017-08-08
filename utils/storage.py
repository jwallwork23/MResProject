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


def csv2table(gauge, setup):
    """
    Convert a .csv timeseries file to a table format.

    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param setup: equation form or mesh resolution used, e.g. 'xcoarse' or 'fine_rotational'.
    :return: a vector x containing points in time and a vector y containing the associated gauge reading values.
    """

    x = []
    y = []
    i = 0
    infile = open('timeseries/{y1}_{y2}.csv'.format(y1=gauge, y2=setup), 'r')
    for line in infile:
        if i != 6:
            i += 1
        elif i == 6:
            xy = line.split(',')
            x.append(xy[0])
            y.append(xy[1])
    return x, y


def plot_gauges(gauge, prob='comparison', log='n'):
    """
    Plot timeseries data on a single axis.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param prob: problem type name string, corresponding to either 'verification' or 'comparison'.
    :param log: specify whether or not to use a logarithmic scale on the y-axis.
    :return: a matplotlib plot of the corresponding gauge timeseries data.
    """

    if prob== 'comparison':
        setup = {0: 'measured_25mins',
                 1: 'xcoarse_25mins',                       # 3,126 vertices
                 2: 'medium_25mins',                        # 25,976 vertices
                 3: 'fine_25mins',                          # 97,343 vertices
                 4: 'anisotropic_point85scaled_rm=30',
                 5: 'goal-based'}
        labels = {0: 'Gauge measurement',
                  1: 'Mesh approach (i)',
                  2: 'Mesh approach (ii)',
                  3: 'Mesh approach (iii)',
                  4: 'Mesh approach (iv)',
                  5: 'Mesh approach (v)'}
    else:
        setup = {0: 'measured',
                 1: 'fine',
                 2: 'fine_rotational',
                 3: 'fine_nonlinear',
                 4: 'fine_nonlinear_rotational'}
        labels = {0: 'Gauge measurement',
                  1: 'Linear, non-rotational equations',
                  2: 'Linear, rotational equations',
                  3: 'Nonlinear, non-rotational equations',
                  4: 'Nonlinear, rotational equations'}
    styles = {0: '-', 1: ':', 2: '--', 3: '-.', 4: '-', 5: '--'}
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    plt.clf()

    # Temporary user specified input for incomplete plots:
    progress = int(raw_input('How far have we got for this gauge? (1/2/3/4/5): ') or 4) + 1

    # Loop over mesh resolutions:
    for key in range(progress):
        val = []
        i = 0
        v0 = 0
        try:
            infile = open('timeseries/{y1}_{y2}.txt'.format(y1=gauge, y2=setup[key]), 'r')
            for line in infile:
                if i == 0:
                    i += 1
                    v0 = float(line)
                val.append(float(line) - v0)
            infile.close()
            if setup[key] in ('fine_nonlinear', 'fine_nonlinear_rotational', 'anisotropic_point85scaled_rm=30',
                              'xcoarse_25mins', 'medium_25mins', 'fine_25mins', 'goal-based'):
                if log == 'n':
                    plt.plot(np.linspace(0, 25, len(val)), val, label=labels[key], linestyle=styles[key])
                else:
                    plt.semilogy(np.linspace(0, 25, len(val)), val, label=labels[key], linestyle=styles[key])
            else:
                if log == 'n':
                    plt.plot(np.linspace(0, 60, len(val)), val, label=labels[key], linestyle=styles[key])
                else:
                    plt.semilogy(np.linspace(0, 60, len(val)), val, label=labels[key], linestyle=styles[key])
        except:
            x, y = csv2table(gauge, setup[key])
            plt.plot(x, y, label=labels[key], linestyle=styles[key])
    plt.gcf()
    if prob == 'comparison':
        plt.legend(bbox_to_anchor=(1.13, 1), loc=1, facecolor='white')  # 'upper right' == 1 and 'lower right' == 4
    else:
        plt.legend(bbox_to_anchor=(1.1, 1), loc=1, facecolor='white')
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')
    if log == 'n':
        plt.savefig('plots/tsunami_outputs/screenshots/full_gauge_timeseries_{y1}_{y2}.png'.format(y1=gauge, y2=prob))
    else:
        plt.ylim((10 ** -1, 10 ** 1))
        plt.savefig('plots/tsunami_outputs/screenshots/log_gauge_timeseries_{y1}_{y2}.png'.format(y1=gauge, y2=prob))
