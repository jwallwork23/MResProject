"""
Module for computing seafloor deformation unp.sing Okada model.

Okada model is a mapping from several fault parameters
to a surface deformation.
See Okada 1985, or Okada 1992, Bull. Seism. Soc. Am.

some routines adapted from fortran routines written by
Xiaoming Wang.


"""

# Variables in this programme
# poisson - Poisson ratio, dimensionless
# no_sf - Number of sub-faults in length and width, dimensionless
# theta - Asymmetry of a np.sinusoidal slip distribution, dimensionless
# okadaparametertext - Okada Parameter File, .txt
#     Focal_Depth - depth of the top of the fault plane, m
#     Fault_Length - length of the fault plane, m
#     Fault_Width - width of the fault plane, m
#     Dislocation - average displacement, m
#     Strike_Direction - angle from North of fault, degrees
#     Dip_Angle - angle from horizontal, degrees
#     Slip_Angle - slip of one fault block compared to another, degrees
#     Fault_Latitude - latitude of top-center of fault plane, degrees
#     Fault_Longitude - longitude of top-center of fault plane, degrees 
# faultgridparametertext - Grid Parameter File, .txt
#     xlower - Western longitude to solve from, degrees.
#     ylower - Southern latitude to solve from, degrees.
#     xupper - Eastern longitude to solve to, degrees.
#     yupper - North latitude to solve to, degrees.
#     mx - number of grid divisions in longitude, dimensionless
#     my - number of grid divisions in latitude, dimensionless

import numpy as np
import string


poisson = 0.3 # Poisson ratio

def getokadaparams (infile):
    # parameter names and values should appear on the same np.single line seperated by a space

    keylist=["Focal_Depth","Fault_Length","Fault_Width","Dislocation","Strike_Direction", \
             "Dip_Angle","Slip_Angle","Fault_Latitude","Fault_Longitude"]

    okadaparams={}
    fid=open(infile,'r')
    keyleft=len(keylist)
    while keyleft> 0 :
        line=string.split(fid.readline())
        if line:
            if line[0] in keylist:
                okadaparams[line[0]]=float(line[1])
                keyleft=keyleft-1
            if line[1] in keylist:
                okadaparams[line[1]]=float(line[0])
                keyleft=keyleft-1

    for key in keylist :
        if not key in okadaparams:
            print('ERROR: parameters for okada fault not fully specified in %s' % (infile))
            exit

    fid.close()
    return okadaparams

def getfaultparams (infile):

    #parameter names and values should appear on the same np.single line seperated by a space

    keylist=["xlower","ylower","xupper","yupper","dx","dy","mx","my"]

    faultgridparams={}
    fid=open(infile,'r')
    keyleft=len(keylist)-2
    while keyleft> 0 :
        line=string.split(fid.readline())
        if line:
            if line[0] in keylist:
                faultgridparams[line[0]]=float(line[1])
                keyleft=keyleft-1
            if line[1] in keylist:
                faultgridparams[line[1]]=float(line[0])
                keyleft=keyleft-1

    faultgridparams['mx'] = int(faultgridparams['mx'])
    faultgridparams['my'] = int(faultgridparams['my'])

    if faultgridparams.has_key('dx')& faultgridparams.has_key('dy'):
        faultgridparams['xupper'] = faultgridparams['xlower'] + faultgridparams['dx']*(faultgridparams['mx']-1)
        faultgridparams['yupper'] = faultgridparams['ylower'] + faultgridparams['dy']*(faultgridparams['my']-1)
    elif faultgridparams.has_key('xupper')&faultgridparams.has_key('yupper'):
        faultgridparams['dx'] = (faultgridparams['xupper']-faultgridparams['xlower'])/(faultgridparams['mx']-1)
        faultgridparams['dy'] = (faultgridparams['yupper']-faultgridparams['ylower'])/(faultgridparams['my']-1)
    else:
        print('ERROR: parameters for fault grid not fully specified in %s' % (infile))
        exit

    for key in keylist :
        if not key in faultgridparams:
            print('ERROR: parameters for fault grid not fully specified in %s' % (infile))
            exit

    fid.close()

    return faultgridparams

def twodgrid(X, Y):
    return [[0]*(len(X)) for y in xrange(len(Y))]

def XYfunc (faultparamfile):

    faultparams = getfaultparams(faultparamfile)
    X=np.linspace(faultparams['xlower'],faultparams['xupper'],faultparams['mx'])
    Y=np.linspace(faultparams['ylower'],faultparams['yupper'],faultparams['my'])

    return X, Y

def faultparamupdate(okadaparamfile,faultparamfile):    
	
    faultparams=getokadaparams(okadaparamfile)
    faultparams.update(getfaultparams(faultparamfile))
	
    return faultparams
 
def faultfunc_length(x, y, dbar, fault_length, faulttype='average', theta=0.5):
    # x should be the current position along strike, or the array of positions along strike
    # dbar should be average displacement
    # fault_length should be the length of the fault in metres
    # faulttype can be either 'average' or 'np.sinusoidal'
    # theta refers to the asymmetry of a np.sinusoidal fault

    faulttype = str(faulttype)
    if faulttype == 'sinusoidal':
        if x<(theta*fault_length):
	    return (np.pi*dbar/2)*np.sin(0.5*np.pi*x/(theta*fault_length))
	elif x>(theta*fault_length):
            return -(np.pi*dbar/2)*np.sin(1.5*np.pi+0.5*np.pi*(x-(theta*fault_length))/(fault_length-(theta*fault_length)))
	elif x == theta*fault_length:
            return (np.pi*dbar/2)
    elif faulttype == 'average':
        return dbar
    # Still needs work
    elif faulttype == 'circular':
        return (np.pi*dbar/2)*np.sin(0.5*np.pi*(x**2+y**2)/(theta*fault_length))

def faultdislgrid(okadaparams):
    
    no_sf = 20. # Number of sub-faults (length and width direction)
    theta1 = 0.35 # Assymmetry of fault (0.5 corresponds to symmetric, 0 & 1 correspond to fully asymmetric.)

    fault_length = okadaparams["Fault_Length"]
    fault_width = okadaparams["Fault_Width"]
    dbar = okadaparams["Dislocation"]

    sflength = float(fault_length)/no_sf
    sfwidth = float(fault_width)/no_sf
    
    Xf = xrange(0, int(fault_length+sflength), int(sflength))
    Yf = xrange(0, int(fault_width+sfwidth), int(sfwidth))

    # Assigning values at each position (not yet averaged)
    Zf = [[faultfunc_length(Xf[k], Yf[l], dbar, fault_length, faulttype='average', theta=theta1) for l in xrange(0, len(Yf))] \
              for k in xrange(0, len(Xf))]

    #Interpolating for each new position (needed to be done for next step where centre is calculated) 
    Xfbar = [(Xf[k]+Xf[k+1])/2. for k in xrange(0, len(Xf)-1)]
    Yfbar = [(Yf[l]+Yf[l+1])/2. for l in xrange(0, len(Yf)-1)]
    Zfbar = [[float((Zf[k][l]+Zf[k][l+1]+Zf[k+1][l]+Zf[k+1][l+1]))/4. for l in xrange(0, len(Yf)-1)] \
		for k in xrange(0, len(Xf)-1)]

    Yfbar = np.array(Yfbar)
    Xfbar = np.array(Xfbar)
    Zfbar = np.array(Zfbar)

    return Xfbar, Yfbar, Zfbar, sflength, sfwidth

def okadamapgrid(okadaparams, X, Y, i, j, Zfbar1=None):
    # Calculates the surface deformation due to a np.single point source
   
    Xfbar, Yfbar, Zfbar, sflength, sfwidth = faultdislgrid(okadaparams)  

    if Zfbar1 != None:
	Zfbar = Zfbar1

    rad = np.pi/180. # conversion factor from degrees to radians
    rr = 6.378e6 # radius of earth
    lat2meter = rr*rad # conversion factor from degrees latitude to meters
    
    hh = okadaparams["Focal_Depth"] # Setting values for each key ie Focal_Depth value is now hh
    th = okadaparams["Strike_Direction"]
    dl = okadaparams["Dip_Angle"]
    rd = okadaparams["Slip_Angle"]
    fault_latitude = okadaparams["Fault_Latitude"]
    fault_longitude = okadaparams["Fault_Longitude"]
    location = okadaparams.get("LatLong_Location", "top center")

    ang_dip = rad*dl #angle of dip in radians
    ang_slip = rad*rd #angle of rake in radians
    ang_strike = rad*th # angle of strike in radians

    sn = np.sin(ang_dip)
    cs = np.cos(ang_dip)
    sn_str = np.sin(ang_strike)
    cs_str = np.cos(ang_strike)
       
    halfsflength = sflength*0.5

    # Convert to the correct depth, lat, lon for the specific sub-fault
    if location == 'top center':
        hh = hh+Yfbar[j]*sn	
	hh += 0.5*sfwidth*sn
	startlat1 = 0
	startlon1 = 0
        if len(Xfbar)%2!=0:
            if i==float((len(Xfbar)-1)/2):
	        y0 = startlat1
	        x0 = startlon1
            elif i>float((len(Xfbar)-1)/2):
	        y0 = startlat1 + Xfbar[i-(len(Xfbar)+1)/2]*cs_str
                y0 += 0.5*sflength*cs_str
	        x0 = startlon1 + Xfbar[i-(len(Xfbar)+1)/2]*sn_str
	        x0 += 0.5*sflength*sn_str
	    elif i<float((len(Xfbar)-1)/2):
                y0 = startlat1 - Xfbar[(len(Xfbar)-1)/2-i]*cs_str
	        y0 += 0.5*sflength*cs_str
                x0 = startlon1 - Xfbar[(len(Xfbar)-1)/2-i]*sn_str
                x0 += 0.5*sflength*sn_str				
        elif len(Xfbar)%2==0:
	    startlat2 = startlat1 + 0.5*sflength*cs_str
            startlon2 = startlat1 + 0.5*sflength*sn_str
	    if i==float(len(Xfbar)/2):
		y0 = startlat2
		x0 = startlon2
	    elif i>float(len(Xfbar)/2):
		y0 = startlat2 + Xfbar[i-(len(Xfbar)+1)/2]*cs_str
		y0 -= 0.5*sflength*cs_str
		x0 = startlon2 + Xfbar[i-(len(Xfbar)+1)/2]*sn_str
		x0 -= 0.5*sflength*sn_str
	    elif i<float(len(Xfbar)/2):
		y0 = startlat2 - Xfbar[(len(Xfbar))/2-i]*cs_str
		y0 += 0.5*sflength*cs_str
		x0 = startlon2 - Xfbar[(len(Xfbar))/2-i]*sn_str
		x0 += 0.5*sflength*sn_str 

	dy = y0-startlat1
	dx = x0-startlon1
	y0 = (1/110.54)*dy*0.001
	currentlat = y0+fault_latitude
	x0 = (1/(111.320*np.cos(currentlat*rad)))*dx*0.001
	currentlon = x0+fault_longitude
	
	del_y = (Yfbar[j]+0.5*sfwidth)*cs*sn_str / lat2meter
	y0 = currentlat - del_y

	del_x = (Yfbar[j]+0.5*sfwidth)*cs*cs_str / (lat2meter*np.cos(y0*rad))
	x0 = currentlon + del_x
	y0 = currentlat - del_y

	d = Zfbar[i][j]
	x,y = np.meshgrid(X,Y)
        
      	# The idea is to get a grid and then stitch together all of the dZs that are outputted
	# in order to form a big Okada initial condition as can be seen with arrayaddition.py
	# where o3.txt+o4.txt = o5.txt

	# Convert distance from (x,y) to (x0,y0) from degrees to meters:
	xx = lat2meter*np.cos(rad*y)*(x-x0)
	yy = lat2meter*(y-y0)
	# Convert to distance along strike (x1) and dip (x2):
	x1 = xx*sn_str + yy*cs_str
	x2 = xx*cs_str - yy*sn_str

	# In Okada's paper, x2 is distance up the fault plane, not down dip:
	x2 = -x2

	p = x2*cs + hh*sn # As specified in Okada '85
	q = x2*sn - hh*cs # As specified in Okada '85

	f1=strike_slip (x1+halfsflength,p, ang_dip,q)
	f2=strike_slip (x1+halfsflength,p-sfwidth,ang_dip,q)
	f3=strike_slip (x1-halfsflength,p, ang_dip,q)
	f4=strike_slip (x1-halfsflength,p-sfwidth,ang_dip,q)

	g1=dip_slip (x1+halfsflength,p, ang_dip,q)
	g2=dip_slip (x1+halfsflength,p-sfwidth,ang_dip,q)
	g3=dip_slip (x1-halfsflength,p, ang_dip,q)
	g4=dip_slip (x1-halfsflength,p-sfwidth,ang_dip,q)
	
	# Displacement in direction of strike and dip:
	ds = d*np.cos(ang_slip)
	dd = d*np.sin(ang_slip)

	us = (f1-f2-f3+f4)*ds
	ud = (g1-g2-g3+g4)*dd
	dZ = (us+ud)
        
    # Incomplete - look at 'top center' for an idea of what needs to be done here
    # to make this programme fully versatile
    # Not tested so may be incorrect.
    elif location == 'centroid':
	if len(Yfbar)%2!=0:
            startdepth = hh+0.5*sfwidth*np.sin(ang_dip)
	    if j==float((len(Yfbar)-1)/2):
	        hh=startdepth
	    elif j>float((len(Yfbar)-1)/2):
		hh = startdepth+Yfbar[j-(len(Yfbar)+1)/2]*np.sin(ang_dip)
		hh += 0.5*sfwidth*np.sin(ang_dip)
	    elif j<float((len(Yfbar)-1)/2):	
		hh = startdepth-Yfbar[(len(Yfbar)-1)/2-j]*np.sin(ang_dip)
		hh += 0.5*sfwidth*np.sin(ang_dip)
	elif len(Yfbar)%2==0:
	    startdepth = hh + sfwidth*np.sin(ang_dip)
	    if j==float(len(Yfbar)/2):
	        hh = startdepth				
	    elif j>float(len(Yfbar)/2):
		hh = startdepth+(Yfbar[j-(len(Yfbar)+1)/2]-0.5*sfwidth)*np.sin(ang_dip)
	    elif j<float(len(Yfbar)/2):
		hh = startdepth-(Yfbar[len(Yfbar)/2-j]-0.5*sfwidth)*np.sin(ang_dip)

    return dZ


def strike_slip (y1,y2,ang_dip,q):
    """
!.....Used for Okada's model
!.. ..Methods from Yoshimitsu Okada (1985)
!-----------------------------------------------------------------------
"""
    
    sn = np.sin(ang_dip)
    cs = np.cos(ang_dip)
    d_bar = y2*sn - q*cs
    r = np.sqrt(pow(y1,2) + pow(y2,2) + pow(q,2))
    a4 = 2.0*poisson/cs*(np.log(r+d_bar) - sn*np.log(r+y2))
    f = -(d_bar*q/r/(r+y2) + q*sn/(r+y2) + a4*sn)/(2.0*3.14159)
    return f


def dip_slip (y1,y2,ang_dip,q):
    """
!.....Based on Okada's paper (1985)
!.....Added by Xiaoming Wang
!-----------------------------------------------------------------------
"""
    
    sn = np.sin(ang_dip)
    cs = np.cos(ang_dip)

    d_bar = y2*sn - q*cs;
    r = np.sqrt(pow(y1,2) + pow(y2,2) + pow(q,2))
    xx = np.sqrt(pow(y1,2) + pow(q,2))
    a5 = 4.*poisson/cs*np.arctan((y2*(xx+q*cs)+xx*(r+xx)*sn)/y1/(r+xx)/cs)
    f = -(d_bar*q/r/(r+y1) + sn*np.arctan(y1*y2/q/r) - a5*sn*cs)/(2.0*3.14159)

    return f

def builddeffile (okadaparamfile,faultparamfile, Zfbar1=None):

    faultparams=getokadaparams(okadaparamfile)
    faultparams.update(getfaultparams(faultparamfile))
    X, Y = XYfunc(faultparamfile)
    Xfbar, Yfbar, Zfbar, sflength, sfwidth = faultdislgrid(faultparams)
    dZtotal=twodgrid(X, Y)

    # Summing together the contribution of each sub-fault to create the
    # effect for the whole fault
    for j in xrange(0, len(Yfbar)):
        for i in xrange(0, len(Xfbar)):
            if Zfbar1==None:
    	        dZ = okadamapgrid(faultparams, X, Y, i, j)
            else:
	        dZ = okadamapgrid(faultparams, X, Y, i, j, Zfbar1)
	    dZtotal = np.array(dZtotal)+np.array(dZ)

    return dZtotal

# Run the functions
def main(Zfbar=None):
    # X, Y, Z are the surface deformation arrays
    # Xfbar, Yfbar, Zfbar are the fault slip arrays

    okadaparametertext1 = 'okada_b2.txt'
    faultgridparametertext = 'fault.txt'

    X, Y = XYfunc(faultgridparametertext)
    Z = builddeffile(okadaparametertext1,faultgridparametertext, Zfbar)

    params = faultparamupdate(okadaparametertext1,faultgridparametertext)
    Xfbar, Yfbar, Zfbar, sflength, sfwidth = faultdislgrid(params)

    return X, Y, Z, Xfbar, Yfbar, Zfbar, sflength, sfwidth
