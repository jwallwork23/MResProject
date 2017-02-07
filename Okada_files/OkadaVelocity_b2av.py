import numpy as np
from math import sqrt
import ast
import Okada_b2av as ok
from Okada_b2av import twodgrid

# Variables in this programme
# vel - Rupture Velocity m/s
# X0, Y0 - Cartesian location of hypocentre on the fault plane, m

def okadavelocity(): 
	
    X, Y, Z, Xfbar, Yfbar, Zfbar, sfl, sfw = ok.main()

    Zfbar2 = twodgrid(Yfbar, Xfbar)

    vel = 4000. # ms-1
    rise_vel = 0.5 # ms-1

    #dt = sfl/vel # Time step

    dt = 2 # roughly what time step you want to use (corrected for later in the programme)
    
    fl = float(max(Xfbar))+0.5*sfl # Fault length
    fw = float(max(Yfbar))+0.5*sfw # Fault width

    # Position of hypocentre [where (0, 0) is the bottom left hand corner]
    X0 = 0.5*fl
    Y0 = 0.5*fw

    # Max time to fully rupture
    endt = 100 # now set to some arbitrary big time that it won't reach
    # Max number of time steps needed
    nt = int((endt/dt) + 1) 
    T = np.linspace(0, endt, nt)


    dist = 0

    times = []
    vels = []
	
    for t in xrange(nt):

        try:
            dist = dist + vel*(T[t+1]-T[t])
        except:
            dist = dist + vel*(T[t]-T[t-1])

        # Safety bit of code to make sure starts at 0 displacement
        if t == 0:
            dist = 0
            disp = 0

        # Safety bit of code to check every part of the fault has ruptured
        if t == T[nt-1] and np.all(Zfbar2) != np.all(Zfbar):
            nt+=1

        # Safety bit of code to cut it off before time, so no programming time is wasted
        if np.all(Zfbar2 == Zfbar):
            break

        Zfbar2dummy = np.copy(np.array(Zfbar2))

        print "Time is %.3f" % (T[t])
        # Loop in which the displacement values are 'turned on' or 'kept off'       
        for i in xrange(len(Xfbar)):
            for j in xrange(len(Yfbar)):
                # If the distance travelled by the rupture front is greater than the distance
                # of the element from the starting point, turn the dislocation on
		if dist >= ((Xfbar[i]-X0)**2+(Yfbar[j]-Y0)**2)**0.5 and Zfbar2[i][j] < Zfbar[i][j]:
                    Zfbar2[i][j] = Zfbar2[i][j] + rise_vel*dt
                elif dist <= ((Xfbar[i]-X0)**2+(Yfbar[j]-Y0)**2)**0.5: # Keep the dislocation off
                    Zfbar2[i][j] = 0.0

        for i in xrange(len(Xfbar)): # making sure that the max. displacement is kept
            for j in xrange(len(Yfbar)):
                if Zfbar2[i][j] > Zfbar[i][j]:
                    Zfbar2[i][j] = Zfbar[i][j]

	# Feeding this new fault dislocation to be solved for surface deformation
        if t != 0:
            Ztotaldummy = np.copy(Ztotal)
            
        X, Y, Ztotal, foo0, foo1, foo2, foo3, foo4 = ok.main(Zfbar2)

        if t != 0:
            try:
                Vtotal = (np.array(Ztotal)-np.array(Ztotaldummy))/(T[t+1]-T[t])
                Vfbar = (np.array(Zfbar2) - np.array(Zfbar2dummy))/(T[t+1]-T[t])
            except:
                Vtotal = (np.array(Ztotal)-np.array(Ztotaldummy))/(T[t]-T[t-1])
                Vfbar = (np.array(Zfbar2) - np.array(Zfbar2dummy))/(T[t]-T[t-1])
        else:
            try:            
                Vtotal = (2*np.array(Ztotal))/(T[t+1]-T[t-1])
                Vfbar = (2*np.array(Zfbar2))/(T[t+1]-T[t-1])
            except:
                try:
                    Vtotal = np.array(Ztotal)/(T[t]-T[t-1])
                    Vfbar = np.array(Zfbar2)/(T[t]-T[t-1])
                except:
                    Vtotal = np.array(Ztotal)/(T[t+1]-T[t])
                    Vfbar = np.array(Zfbar2)/(T[t+1]-T[t])
       
        vels.append(Vtotal)
        times.append(T[t])

    X = np.array(X)
    Y = np.array(Y)
    times = np.array(times)
    vels = np.array(vels)
    return X, Y, times, vels

def velocity_of_surface(lonlat, time, X, Y, T, V):
    # The time steps used must be exactly the same as the time steps used in Fluidity. (recommended 2 seconds)
    # lonlat should be a tuple of at least (lon, lat) [Fluidity produces (lon, lat, radius)]
    # time should be a float of the current time step
    # This function returns the surface velocity at any location in time.

    lon_2_interp = lonlat[0]
    lat_2_interp = lonlat[1]
    T = list(T)
    t = T.index(time)
    Ylen = len(Y)-1
    Xlen = len(X)-1

    lat_2_index = Ylen*(1-(Y[-1]-lat_2_interp)/(Y[-1]-Y[0]))
    lon_2_index = Xlen*(1-(X[-1]-lon_2_interp)/(X[-1]-X[0]))

    latitudes = np.linspace(0, Ylen, V.shape[1])
    longitudes = np.linspace(0, Xlen, V.shape[2])

    # If it's on the first iteration read everything in, this may be the thing taking up the time.    
    V1 = V[t,:,:] # brought down to 2D without the need to interpolate
    V2 = [] # to be brought down to 1D by interpolation
    append2 = V2.append
    for lons in longitudes:
        append2(np.interp(lat_2_index, latitudes, V1[:,lons]))

    surfvel = np.interp(lon_2_index, longitudes, V2)
    
    return surfvel
print okadavelocity()
