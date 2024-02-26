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

This module contains all the functionalities to do the image processing in
three steps:
    - Thresholding via Otsu's method (More details in OpenCV documentation)
    - Filtering via morphological opening operation with a line kernel
    - Midline Extraction by taking the median value at each discrete x-value 
      of the filtered image
    
"""

import cv2 
import numpy as np
import bendstiff

plot_par = [] # Initialize plot parameters list for plotting
        
def find_coord(img):
    """
    This function finds the coordinates of the deflection curve in a binary 
    image

    Parameters
    ----------
    img : array
        Binary image.

    Returns
    -------
    coord : numpy array
        Array containing the x and y coordinates of the image.

    """
    # Function for finding coordinates in a binary image
    x = []; y = []
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img[j][i] == 255:
                x.append(i)
                y.append(-j)
    coord    = [x,y]
    coord    = np.array(coord)
    coord[1] = coord[1] - coord[1,0] # Make deflection start in 0
    return coord

def median_midline_extraction(img):
    """
    Function for extracting the midline of the deflection curve by finding the
    median y value at each discrete x value.
    
    Parameters
    ----------
    img : array
        Binary image.

    Returns
    -------
    coord : numpy array
        Array containing the x and y coordinates of the image.

    """
    x = [] 
    y = []
    for i in range(img.shape[1]):
        hv = np.where(img[:,i]==255)
        y.append(-np.median(hv)) if len(hv[0]) > 0 else y.append(float('nan'))
        x.append(i+1)
    coord    = [x,y]
    coord    = np.array(coord)
    hv2      = np.argwhere(np.isnan(coord[1])) # Find nan entries in 
    coord    = np.delete(coord, hv2, axis=1)
    coord[1] = coord[1] - coord[1,0] # Make sure that deflection start in 0
    
    # Functionality for removing the right edge.
    rem_edge = []
    for i in range(2*coord.shape[1] // 3, coord.shape[1]):
        if coord[1,i-1] < coord[1,i]:
            rem_edge.append(i)
        elif coord[1,i-4] < coord[1,i]:
            rem_edge.append(i)
    coord = np.delete(coord, rem_edge, 1)
    return coord

def line_kernel(length, angle):
    """
    Function for generating line kernel 

    Parameters
    ----------
    length : float
        Length of the kernel.
    angle : float
        Angle of the kernel in radians.

    Returns
    -------
    kernel : numpy array
        Line kernel.

    """
    kernelx = round(abs(np.cos(angle))*length)+1
    kernely = round(abs(np.sin(angle))*length)+1 
    
    if kernelx == 1: return np.ones([kernely,kernelx], dtype = 'uint8') 
    
    kernel = np.zeros([kernely,kernelx], dtype = 'uint8')
    alpha  = np.sign(angle)*np.arctan((kernely-1)/(kernelx-1))
    
    for n in range(max(kernelx,kernely)): 
        if kernelx >= kernely:
            i = n
            d = - round(i * np.tan(alpha))  
            j = d-1 if angle > 0 else d
        else:
            j = n
            d = - round(j * 1/np.tan(alpha))  
            i = d-1 if angle > 0 else d
        kernel[j][i] = 1
    return kernel

def thresholding(img):
    """
    Automatic tresholding of image using Otsu critera

    Parameters
    ----------
    img : array
        Grey-scale image.

    Returns
    -------
    img_th : array
        Binary image.

    """
    th, img_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img_th

def smooth(img_th, par):
    """
    Function for smoothing the binary image using morhpologcial filters

    Parameters
    ----------
    img_th : array
        Binary image that are to be smoothed.
    par : dict
        Parameters used for the analysis.

    Returns
    -------
    img : array
        Smoothed image.

    """
    img = np.copy(img_th)
    
    ##### Fifth-order polynomial fit using the in-house spline fit #####
    # Find datapoints to use in the spline fit 
    coord = find_coord(img) 
    x = coord[0] 
    y = coord[1]

    # Constraints on the fit
    con = ({'x':np.max(x), 'f(x)':0.0 , 'der':2 }, 
           {'x':np.min(x), 'f(x)':0.0 , 'der':1 },)
    
    poly, B, omega_jk = bendstiff.curvefit.smoothing_spline(x, y, 1.0, m=3,    \
                             thin = True, con = con, segments = 2)
    
    # Calculate spline values (for plotting)
    x_pf = np.linspace(0 , np.max(x), 200)
    y_pf = np.array([bendstiff.bsplines.spline_calc(a, poly) for a in x_pf]) 
    y_pfd = np.array([bendstiff.bsplines.spline_calc(a, poly, der=1)           \
                      for a in x_pf]) 
    
    # Make kernels (structuring elements) based on the slope of curve
    da = np.min(y_pfd) / (par['n_r'] - 1) 
    a = [da*i for i in range(par['n_r'])]
    b = [a[i] - da/2 for i in range(1,par['n_r'])] 
    
    # Find the boundaries of each region
    idx = []
    xb = [0]
    for i in range(len(b)): 
        idx.append(np.where(y_pfd <= b[i])[0][0])
        xb.append(int(x_pf[idx[i]]))
    xb.append(img[0].size) 
    
    # Smooth each region using the line kernel with openeing operation
    size_kernel = int((xb[1] - xb[0])/par['L_kernel']) 
    for i in range(len(a)):
        kernel = line_kernel(size_kernel, np.arctan(a[i])) 
        img_part = img[:, xb[i]:xb[i+1]]
        img_part = cv2.morphologyEx(img_part, cv2.MORPH_OPEN, kernel)
        img[:, xb[i]:xb[i+1]] = cv2.morphologyEx(img_part, cv2.MORPH_CLOSE,    \
                                                 kernel)
    
    if par['plot'] == True:
        plot_par.append( {"Title": 'Fitted 5th order polynomial', "Order": 3, 
                          "Type": 'scatter', "Data": (x,y),
                          'Label': 'Data points'} )
        plot_par.append( {"Title": 'Fitted 5th order polynomial', "Order": 3, 
                          "Type": 'xy', "Data": (x_pf,y_pf),
                          'Label': 'Curve fit'} )
        plot_par.append( {"Title": 'Derivative of fitted 5th order polynomial', 
                          "Order": 4, "Type": 'xy', "Data": (x_pf,y_pfd)} )
        plot_par.append( {"Title": 'Smoothened image', "Order":2, 
                          "Type": 'image', "Data": img} )
    return img

def run_improc(img_name, par):
    """
    Function for running the image processing module.

    Parameters
    ----------
    img_name : array
        Raw grey-scale image.
    par : dict
        Parameters used for the analysis.

    Returns
    -------
    x : numpy array
        Vector of x-values of the deflection curve.
    y : numpy array
        Vector of y-values of the deflection curve.

    """
    
    # Read the image
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE);
        
    # Automatic tresholding of image using Otsu critera
    img_th = thresholding(img)
    
    # Smooth the image
    img_smooth = smooth(img_th, par) if par['smooth'] == True else img_th 
    
    # Extract the coordinates of the midline
    coord = median_midline_extraction(img_smooth)
    
    # Convert coordinates to mm and place at x = 0mm
    x = coord[0] 
    y = coord[1]
    x = x - np.min(x) # Place at x = 0 # Might be redundant
    x = par['mpp']*x  # Convert from pixels to mm
    y = par['mpp']*y  # Convert from pixels to mm
    
    # Choose fewer amount of datapoints.
    x, y = bendstiff.utility.thin_xy_data(x, y, par['n'])
    
    if par['plot'] == True:
        plot_par.append( {"Title": 'Original image', "Order": 0 , 
                          "Type": 'image', "Data": img} )
        plot_par.append( {"Title": 'Thresholded image', "Order": 1 , 
                          "Type": 'image', "Data": img_th} )
        plot_par.append( {"Title": 'Fabric deflection', "Order": 5 , 
                          "Type": 'scatter', "Data": (x,y),'xlabel':'x [mm]',
                          'ylabel':'y [mm]'} )
        
        # Create subplot figures with title and nrows,ncols
        fig,axes = bendstiff.utility.subplot_creator('Image Processing', 2, 3, \
                                                     figsize = (15, 10))
        
        # Fill out the subplots
        bendstiff.utility.subplot_filler(axes,plot_par)
        
        if 'results_dir' in par:
            fig.savefig(par['results_dir']+'/image_processing.pdf')
        
    return x, y


    
    
    
    
