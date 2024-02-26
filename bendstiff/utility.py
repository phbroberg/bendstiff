# -*- coding: utf-8 -*-
"""
############################################################################
#  This Python file is part of bendstiff for calculating the moment-       #
#  curvature relation for textile materials.                               #
#                                                                          #
#  The code was developed at Department of Materials and Production at     #
#  Aalborg University by  P.H. Broberg, E. Lindgaard, C. Krogh,            #
#  S.M. Jensen, G.G. Trabal, A.F.-M Thai, B.L.V. Bak                       #
#                                                                          #
#  A github repository, with the most up to date version of the code,      #
#  can be found here:                                                      #
#     https://github.com/phbroberg/bendstiff                               #
#                                                                          #
#  The code is open source and intended for educational and scientific     #
#  purposes only. If you use this script in your research, the developers  #
#  would be grateful if you could cite the paper.                          #
#                                                                          #
#  Disclaimer:                                                             #
#  The authors reserve all rights but do not guarantee that the code is    #
#  free from errors. Furthermore, the authors shall not be liable in any   #
#  event caused by the use of the program.                                 #
############################################################################

Module containing utility functions used throughout the software. 

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

def thin_xy_data_power(x, y, K, power):
    """
    Utility function used to thin xy-data, to have K entries following a 
    nonlinear sampling based on the power.

    Parameters
    ----------
    x : array
        x-values.
    y : array
        y-values. 
    K : int
        Number of values to sample.
    power : float
        Factor used to describe the sampling power.

    Returns
    -------
    x_thin : array
        x-values after thinning.
    y_thin : array
        y-values after thinning.

    """
    idx = np.linspace(0,x.size-1,K,dtype = int) #Try
    
    idx = (idx/max(idx))**power * max(idx)
    
    idx = idx.astype(int)
    for i in range(len(idx)):
        if any(idx[i] <= idx[:i]): idx[i] = idx[i-1]+1
    x_thin = x[idx]; y_thin = y[idx]
    return x_thin, y_thin

def thin_xy_data(x, y, K):
    """
    Utility function used to thin xy-data, to have K entries following a 
    linear sampling 
    
    Parameters
    ----------
    x : array
        x-values.
    y : array
        y-values. 
    K : int
        Number of values to sample.

    Returns
    -------
    x_thin : array
        x-values after thinning.
    y_thin : array
        y-values after thinning.

    """
    # Utility function used to thin xy-data, to have K entries
    idx = np.linspace(0,x.size-1,K,dtype = int) #Try
    
    x_thin = x[idx]; y_thin = y[idx]
    return x_thin, y_thin

def linearise_first_part(moment, curvature):
    """
    Function for linearising the first part of the moment-curvature relation in
    case of oscillations at low curvatures. The bending stiffness based on the
    secant stiffness is also calculated to be inputted in a simulator. 

    Parameters
    ----------
    moment : array
        Moment vector.
    curvature : array
        Curvature vector.

    Returns
    -------
    moment : array
        Modified moment vector.
    curvature : array
        Modified curvature array.
    bending_stiffness : array
        Bending stiffness based on the secant stiffness.

    """
    # Utility function used to linearise first part of curve
    curvature_diff = np.diff(curvature)
    moment_diff = np.diff(moment)
    tan_stiff = moment_diff / curvature_diff
    sec_stiff = moment / curvature
    idx = 3
    kappa_0 = 0-1
    while tan_stiff[idx] < sec_stiff[idx+1] and tan_stiff[idx] > 0 and         \
                             curvature[idx] > 0 and idx < len(curvature_diff):
        idx += 1
    curvature = curvature[:(idx-kappa_0)+1]; moment = moment[:(idx-kappa_0)+1]
    curvature[idx-kappa_0] = 0.0; moment[idx-kappa_0] = 0.0
    
    ###### Calculate bending stiffness ######
    bending_stiffness = moment[:idx-kappa_0] / curvature[:idx-kappa_0]
    bending_stiffness = np.append(bending_stiffness,                           \
                                  bending_stiffness[idx-kappa_0-1])
    #bending_stiffness = [curvature[:-1],np.diff(moment) / np.diff(curvature)]
    return moment, curvature, bending_stiffness

def is_pos_def(x):
    """
    Functionality for checking if an array is positive definite

    Parameters
    ----------
    x : array
        Array to be checked if positive definite.

    Returns
    -------
    bool
        True if positive definite.

    """
    # Check if a matrix is positive definite
    return np.all(np.linalg.eigvals(x) > 0)

def log_interp(zz, xx, yy):
    """
    Functionality for making logharithmic interpolation of data

    Parameters
    ----------
    zz : array
        Values to interpolate
    xx : array
        x data points
    yy : array
        y data points

    Returns
    -------
    array
        interpolated values

    """
    # Logarithmic interpolation. Used for knot thinning strategy.
    logz = np.log10(zz); logx = np.log10(xx); logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))


def subplot_creator(suptitle, rows, cols, figsize = None):
    """
    Function for generating the subplots.

    Parameters
    ----------
    suptitle : str
        Title of the figure.
    rows : int
        Number of rows in the figure.
    cols : int
        Number of coloumns in the figure.

    Returns
    -------
    fig : figure object
        Figure object
    axes : array
        Initialised array for containing plot parameters

    """

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize = figsize)
        
    fig.suptitle(suptitle, fontsize=16)
    
    return fig, axes

def subplot_filler(axes, plot_par, results_dir = ''):
    """
    Function for filling the figure.

    Parameters
    ----------
    axes : array
        initial array for plotting
    plot_par : list
        List with content to be filled in the plot

    Returns
    -------
    None.

    """
    
    # Flatten axes for ease of looping
    axf     = axes.flatten()
    
    # Sort the plot_par list according to the order of figures specified in dictionary
    newlist = sorted(plot_par, key=lambda k: k['Order'])
    
    # Extract the order list
    values_of_key = [dict_par['Order'] for dict_par in newlist]

    # Loop over figures
    for i, j in enumerate(values_of_key):
        # print('This is for newlist',i)
        # print('This is for axf',j)
        if newlist[i]['Type'] == 'image':
            axf[j].imshow(newlist[i]['Data'],'gray'); 
            axf[j].title.set_text(newlist[i]['Title'])
            
        if newlist[i]['Type'] == 'scatter':
            if 'Label' in newlist[i]:
                axf[j].scatter(newlist[i]['Data'][0],newlist[i]['Data'][1],    \
                               s = 5, color = 'r',label=newlist[i]['Label'])
                axf[j].legend()
            else:
                
                axf[j].scatter(newlist[i]['Data'][0],newlist[i]['Data'][1],    \
                               s = 5, color = 'r'); 
            axf[j].title.set_text(newlist[i]['Title'])
            
            
        if newlist[i]['Type'] == 'xy':
            if 'Label' in newlist[i]:    
                axf[j].plot(newlist[i]['Data'][0],newlist[i]['Data'][1],       \
                            alpha=0.5,label=newlist[i]['Label']); 
                axf[j].legend()
            else: 
                axf[j].plot(newlist[i]['Data'][0],newlist[i]['Data'][1],       \
                            alpha=0.5); 
                
            axf[j].title.set_text(newlist[i]['Title'])
            if 'xlabel' in newlist[i]:
                axf[j].set_xlabel(newlist[i]['xlabel'])
                axf[j].set_ylabel(newlist[i]['ylabel'])
        
def error(x, y, xa, ya, n):
    """
    Function for calculating the errors between two data sets. 

    Parameters
    ----------
    x : array
        True x-values.
    y : array
        True y-values.
    xa : array
        Approximated x-values
    ya : array
        Approximated y-values
    n : int
        Number of points to calculate the error.

    Returns
    -------
    abs_error : array
        The absolute error at each point
    relative_error : array
        The relative error at each point.
    average_error : float
        The average relative error. 

    """
    xe = np.linspace(max(min(x),min(xa)), min(max(x),max(xa)), n) # Create values
    ye = np.interp(xe, x, y); yae = np.interp(xe, xa, ya)
    
    abs_error = abs(ye[1:] - yae[1:])
    relative_error = abs_error/abs(ye[1:])
    
    average_error = sum(relative_error)/(n-1)
    
    return abs_error, relative_error, average_error
            
def save_par(test_name, W_a, w, p, mpp, m=4, vl=0, smooth=True, plot=False,
             n=50, n_r = 8, L_kernel = 0.5, con = (), results_dir = ''):
    """
    Function for saving the parameters used for the analysis and initialising
    default parameters.

    Parameters
    ----------
    test_name : str
        Name of the image to analyse.
    W_a : float
        Areal load of the fabric.
    w : float
        Width of the fabric specimen.
    p : float
        Value of the smoothing parameter.
    mpp : float
        Metres to pixel conversion factor.
    m : int, optional
        Order if the smoothing spline. The default is 4.
    vl : float, optional
        Vertical load placed at the tip of the specimen. The default is 0.
    smooth : bool, optional
        Specify wether smoothing should be done. The default is True.
    plot : bool, optional
        Specify if plots from each step should be showed. The default is False.
    n : int, optional
        Number of data points extracted from the image processing. The default 
        is 50.
    n_r : int, optional
        Number of regions used in the smoothening. The default is 8.
    L_kernel : float, optional
        Length of the line kernel used for the smoothening. Defined as a factor
        of the length of the first region. The default is 0.5.
    con : dict, optional
        Constraints used for the smoothing spline fit. The default is ().
    results_dir : str, optional
        Name of the folder for storing images. The default is ''.

    Returns
    -------
    par : dict
        Dictionary containing all the parameters used for the analysis.

    """
    # Define dictionary with parameters for easy transfer between functions
    par = {'W_a':W_a, 'w':w, 'mpp':mpp, 'm':m, 'p':p, 'vl':vl, 'smooth':smooth,\
           'plot':plot, 'n':n, 'n_r':n_r, 'L_kernel':L_kernel, 'con':con,      \
           'results_dir':results_dir}
    dict_file = open(test_name+'.pkl', 'wb')
    pickle.dump(par, dict_file)
    dict_file.close()
    return par

def load_par(test_name):
    """
    Function for loading parameters from a previous analysis

    Parameters
    ----------
    test_name : str
        Name of the test.

    Returns
    -------
    par : dict
        Parameters to be used by the analysis.

    """
    par_file = open(test_name+".pkl", "rb")
    par = pickle.load(par_file)
    par_file.close()
    return par
