def wrapper(func, *args, **kwargs) :
    """A wrapper function to enable timing of functions with arguments"""
    def wrapped() :
        return func(*args, **kwargs)
    return wrapped



def gauge_timeseries(gauge, dat) :
    """A function for storing timeseries data for a particular gauge."""

    name = raw_input('Enter a name for this time series: ')
    outfile = open('timeseries/{y1}_{y2}.txt'.format(y1 = gauge, y2 = name), 'w+')

    for i in range(len(dat)) :
        outfile.write(str(dat[i]) + '\n')

    outfile.close()