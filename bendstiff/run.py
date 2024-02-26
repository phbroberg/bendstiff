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

This module contains funcitonalities for executing the one-click bending
stiffness.

"""

import bendstiff
import numpy as np
import pandas as pd

plot_par = [] # Initialize plot parameters list for plotting

def run_bendstiff(image_name, par, validation = False):
    """
    Function for running the one-click bending stiffness.

    Parameters
    ----------
    image_name : str
        Raw image.
    par : dict
        Parameters used for the analysis.
    validation : bool, optional
        Specify wether the results should be compared with a simulation. The 
        default is False. Setting this true requires you to have data for
        deflection curve and/or a moment-curvature relation to compare with 
        the computed values.

    Returns
    -------
    x : array
        x-values of the deflection curve.
    y : array
        y-values of the deflection curve.
    curvature : array.
        Curvature vector
    moment : array
        Moment vector.

    """
    # Run the image processing module
    x, y = bendstiff.image_processing.run_improc(image_name, par) 
    
    # Modify constraints according to maximum x value
    for i in range(len(par['con'])):
        if par['con'][i]['x'] == 'x_max': 
            par['con'][i]['x'] = np.max(x)               
    
    # Make the curve fit
    spf = bendstiff.curvefit.run_curvefit(x, y, par)
        
    # Calculate the moment and the curvature
    moment, curvature, arc_length = bendstiff.mocurv.run_mocurv(x, y, spf, par)
    
    ###### Modify moment-curvature relation to make initial part linear ######
    moment_ori = np.copy(moment) 
    curvature_ori = np.copy(curvature)
    moment, curvature, bending_stiffness =                                     \
        bendstiff.utility.linearise_first_part(moment, curvature)

    if validation == True: 
        bendstiff.run.run_validation(moment, curvature, moment_ori,            \
                                     curvature_ori, bending_stiffness, x, y)

    return x, y, curvature, moment


def run_validation(moment, curvature, moment_ori, curvature_ori, 
                   bending_stiffness, x, y):
    """
    Function for plotting the computed deflection curve and moment-curvature 
    relation with a simulation to validate the method.

    Parameters
    ----------
    moment : array
        Moment vector.
    curvature : array
        Curvature vector. 
    moment_ori : array
        Moment vector before it is modified.
    curvature_ori : array
        Curvature vector before it is modified
    bending_stiffness : array
        Secant bending stiffness at each point.
    x : array
        x-values of the deflection curve.
    y : array
        y-values of the deflection curve.

    Returns
    -------
    None.

    """
    
    fig,axes = bendstiff.utility.subplot_creator('Validation',2,2)

    # Modified moment curvature
    plot_par.append({"Title": 'Moment-curvature', "Order": 0, "Type": 'xy', 
                     "Data": (curvature,moment),'Label':'Modified'})
    plot_par.append({"Title": 'Moment-curvature', "Order": 0, "Type": 'xy',
                     "Data": (curvature_ori,moment_ori),'Label':'Original',
                     'xlabel': '$\kappa$ [mm$^{-1} $]', 'ylabel':'$M$ [N]' })


    plot_par.append({"Title": 'Bending stiffness', "Order": 1, "Type": 'xy', 
                     "Data": (curvature,bending_stiffness)})

    ####### Compare with model ###########
    load_data = pd.read_excel (r'simulator_results.xlsx') 
    data = np.array(load_data)
    x_sim = data[:,0]
    y_sim = data[:,1] 
    curvature_comp = data[:,2]
    moment_comp = data[:,3]

    # Compare moment-curvature
    plot_par.append({"Title": 'Compare moment-curvature', "Order": 2, 
                     "Type": 'xy', "Data": (curvature_comp,moment_comp),
                     'Label':'Simulator','xlabel': '$\kappa$ [mm$^{-1} $]', 
                     'ylabel':'$M$ [N]' })
    plot_par.append({"Title": 'Compare moment-curvature', "Order": 2, 
                     "Type": 'scatter', "Data": (curvature,moment),
                     'Label':'Measured','xlabel': '$\kappa$ [mm$^{-1} $]', 
                     'ylabel':'$M$ [N]' })

    # Compare deflection
    plot_par.append({"Title": 'Compare deflection', "Order": 3, "Type": 'xy', 
                     "Data": (x_sim,y_sim), 'Label':'Simulation',
                     'xlabel': '$x$ [m]', 'ylabel':'$y$ [m]' })
    plot_par.append({"Title": 'Compare deflection', "Order": 3, 
                     "Type": 'scatter', "Data": (x,y), 'Label':'Measured',
                     'xlabel': '$x$ [m]', 'ylabel':'$y$ [m]'})

    # Fill out the subplots
    bendstiff.utility.subplot_filler(axes,plot_par)


    
    
    