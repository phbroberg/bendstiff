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

This module contains functions for calculating the curvature and the moment
distribution of a spline function of a cantilevered beam.

"""
import bendstiff.utility
import matplotlib.pyplot as plt
import numpy as np

plot_par = [] # Initialize plot parameters list for plotting

def M(s, x, par, arc_length_list):
    """
    Function for calculating the moment at a given point

    Parameters
    ----------
    s : int
        Index at which the moment is calculated.
    x : array
        x-values of the deflection curve
    par : dict
        Parameters used for the analysis.
    arc_length_list : list
        List containing the arc length of each segment.

    Returns
    -------
    float
        Moment at the s'th point.

    """
    q = par['W_a']*par['w']
    moment = 0.0
    arm_vl = np.max(x) - x[s] # Arm for the vertical load
    for i in range(s, x.size-1):
        arm = ((x[i]-x[s]) + (x[i+1]-x[s])) / 2  
        weight = q*arc_length_list[i] 
        moment = moment + arm*weight
    return moment + arm_vl*par['vl']

def calc_arc_length(x,y):
    """
    Calculating the arc length

    Parameters
    ----------
    x : array
        x-values of the deflection curve.
    y : array
        y-values of the deflection curve.

    Returns
    -------
    arc_length : float
        Total length of the curve.
    arc_length_list : list
        List containing length of each segment.

    """
    arc_length = 0.0
    arc_length_list = []
    for i in range(x.size-1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        arc_length = arc_length + np.sqrt(dx**2 + dy**2)
        arc_length_list.append(np.sqrt(dx**2 + dy**2))
    return arc_length, arc_length_list

def analytical_linear_moment(x, par):
    """
    Function for calculating the moment distribution of a cantilevered specimen
    with constant distributed loading, using small deformations assumptions.

    Parameters
    ----------
    x : array
        x-values of the deflection curve.
    par : dict
        Parameters for the analysis.

    Returns
    -------
    x_am : array
        x-values for the analytical moment distribution.
    M_am : array
        Moment for the analytical moment distribution.

    """
    x_am = np.linspace(0, np.max(x), 100)
    q = par['W_a']*par['w']
    M_am = q/2 * (x_am**2 - 2*np.max(x)*x_am + np.max(x)**2)     
    return x_am, M_am

def pierce_cantilever(x, y, length, par):
    """
    Function for calculating a constant bending stiffness based on peirce
    cantilever test.

    Parameters
    ----------
    x : array
        x-values of the deflection curve.
    y : array
        y-values of the deflection curve.
    length : float
        Bending length of the deflection curve.
    par : dict
        Parameters used for the analysis.

    Returns
    -------
    G : float
        Constant bending stiffness.

    """
    theta = np.arctan(-np.min(y)/np.max(x))
    S = np.power(length, 3) / 8 * np.cos(theta/2) / np.tan(theta)
    G = S * par['W_a'] #/ 1000000
    return G # Nmm

def run_mocurv(x, y, spf, par):
    """
    Function for running the moment and curvature calculation of the 
    cantilevered specimen.

    Parameters
    ----------
    x : array
        x-values of the deflection curve.
    y : array
        y-values of the deflection curve.
    spf : tuple
        Spline parameters obtained from the spline fit.
    par : dict
        Parameters used for the analysis.

    Returns
    -------
    moment_vector : array
        Moment at each discrete point
    y_spc : array
        Curvatures at each discrete point
    arc_length_list : array
        List containing the arc length of each segment

    """
    # Calculate discrete spline points
    x_sp = np.linspace(0 , np.max(x), 100)
    y_sp = np.array([bendstiff.bsplines.spline_calc(a, spf) for a in x_sp]) 
    
    # Calculate arc length of the spline
    arc_length, arc_length_list = calc_arc_length(x_sp,y_sp)
    
    # Calculate spline curvature     
    sec_der = np.array([bendstiff.bsplines.spline_calc(a, spf, der = 2) for a  \
                        in x_sp]) 
    fir_der = np.array([bendstiff.bsplines.spline_calc(a, spf, der = 1) for a  \
                        in x_sp]) 
    y_spc = - sec_der / np.power((1+fir_der**2), 1.5)
    
    # Calculate the moment for each point
    moment_vector = np.zeros(x_sp.size)
    for k in range(x_sp.size):
        moment_vector[k] = M(k, x_sp, par, arc_length_list)
    
    # Calculate the analytical linear moment for comparison with measured 
    x_am, M_am = analytical_linear_moment(x_sp, par)
    
    # Make moments independent of width
    moment_vector = moment_vector / (par['w']) # *1000
    M_am = M_am / (par['w']) #*1000
    
    # Plot the moment wrt. curvature comnpared with the Peirce cantilever 
    x_pct = np.linspace(0,np.max(y_spc))
    pct = pierce_cantilever(x, y, arc_length, par)
    y_pct = pct  * x_pct # In N to make independent of thickness * 50
    print('Result from Pierce cantilever test: {:1.6f}'.format(pct))
    
    if par['plot'] == True:
        
        fig,axes = bendstiff.utility.subplot_creator('Moment-Curvature', 1, 2, \
                                                     figsize = (10, 5))
        
        plot_par.append( {"Title": 'Moment', "Order": 0, "Type": 'xy', 
                          "Data": (x_sp,moment_vector), 'xlabel': 'x [mm]', 
                          'ylabel':'$M$ [N]', 'color': 'b',
                          'Label': 'Experimental moment distribution'} )
        plot_par.append( {"Title": 'Moment'           , "Order": 0, 
                          "Type": 'xy', "Data": (x_am,M_am), 
                          'xlabel': 'x [mm]', 'ylabel':'$M$ [N]',
                          'Label': 'Analytical moment distribution' })
        plot_par.append( {"Title": 'Bending stiffness', "Order": 1, 
                          "Type": 'xy', "Data": (y_spc,moment_vector),
                          'xlabel': '$\kappa$ [m$^{-1} $]', 'ylabel':'$M$ [N]', 
                          'Label': 'Non-constant bending stiffness'})
        plot_par.append( {"Title": 'Bending stiffness', "Order": 1, 
                          "Type": 'xy', "Data": (x_pct,y_pct), 
                          'xlabel': '$\kappa$ [m$^{-1} $]', 'ylabel':'$M$ [N]', 
                          'Label': 'Constant bending stiffness'})
        
        # Fill out the subplots
        bendstiff.utility.subplot_filler(axes,plot_par)
        if 'results_dir' in par:
            fig.savefig(par['results_dir']+'/mocurv.pdf')
        
    return moment_vector, y_spc, arc_length_list