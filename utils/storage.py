import matplotlib.pyplot as plt
from matplotlib import rc


def gauge_timeseries(gauge, dat) :
    """A function for storing timeseries data for a particular gauge."""

    name = raw_input('Enter a name for this time series: ')
    outfile = open('timeseries/{y1}_{y2}.txt'.format(y1 = gauge, y2 = name), 'w+')

    for i in range(len(dat)) :
        outfile.write(str(dat[i]) + '\n')

    outfile.close()



def plot_gauges(gauge) :
    """A function for plotting timeseries data on a single axis."""

    curves = {1 : coarse, 2 : fine, 3 : simple, 4 : adjoint}
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')

    for key in curves :

        val = []
        infile = open('timeseries/{y1}_{y2}.txt'.format(y1 = gauge, y2 = curves[key]), r)

        for line in infile :
            val.append(float(line))

        infile.close()

        # Plot gauge time series:
        plt.plot(np.linspace(0, 60, len(val)), val)

    plt.gcf().subplots_adjust(bottom = 0.15)
    plt.ylim([-5, 5])
    plt.legend()
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')
    plt.savefig('plots/tsunami_outputs/screenshots/full_gauge_timeseries_{y}.png'.format(y = gauge))



def wrapper(func, *args, **kwargs) :
    """A wrapper function to enable timing of functions with arguments"""
    def wrapped() :
        return func(*args, **kwargs)
    return wrapped