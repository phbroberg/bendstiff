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

This module contains the functionalities to compute B-spline basis functions
and B-spline basis function derivatives. The algorithms used in this script
are based on the ones provided by [1].

References
-----------
[1] Piegl, L and Tiller, W. Monographs in Visual Communication, 1997

"""

import numpy as np
import bendstiff

def knots(x, p, thin=False, segments = 0):
    """
    Function for defining the knot vector for the spline. As smoothing splines
    are used the x-data is used as the knot placement. 
    
    To increase computational efficiency this function can reduce the number of 
    knots using a thinning algorithm. The thinning algorithm for the knots are 
    based on the methodology used R in the smooth.spline() function. 
    The algorithm is defined in [2].
    
    Parameters
    ----------
    x : numpy array
        x-values data points
    p : int
        Order of the spline.
    thin : bool, optional
        Bool variable to determine if the thinning algorithm should be called. 
        The default is False.
    segments : int, optional
        If this parameter is defined, the knot vector is defined by number of 
        segments. This should not be defined when using smoothing splines. 
        The default is 0 (knot vector defined by x-values).

    Returns
    -------
    t : numpy array
        Knot vector.
        
    """
    
    if segments > 0: # Inserted to make variable length of the knots
        K = segments
        t = []
        for k in range(K): t.append(np.quantile(x, k/(K-1)))
        for i in range(p):
            t.insert(0, float(np.min(x)))
            t.append(float(np.max(x)))
        return t
        
    if thin == True: # Thinning algorithm
        n = len(x)
        if n < 50:
            K = n
        elif n <= 3200:
            K = round(bendstiff.utility.log_interp(n, [50,200,800,3200],       \
                                                   [50,100,140,200]))
        else:
            K = round(200 + (np.power(n-3200, 1/5.)))
        t = []
        for k in range(K): t.append(np.quantile(x, k/(K-1)))
    else: 
        t = x.copy()
    
    for i in range(p): # Repeat boundary knots
        t.insert(0, np.min(x))
        t.append(np.max(x))
    return t



def find_span(t,p,T):
    """
    Function for determining the knot span index
    Based on algorithm A2.1 from [1]
    
    Parameters
    ----------
    t : float
        Value of parameter
    p : int
        Order of the spline
    T : list
        Knot vector

    Returns
    -------
    mid : int
        Span index.
        
    """
    n = (len(T)-1) - p - 1
    if t == T[n+1]: # Special case
        return n
    low = p; high = n+1
    mid = int((low+high)/2)
    while (t < T[mid]) or (t >= T[mid+1]):
        if t < T[mid]:
            high = mid
        else:
            low = mid
        mid = int((low + high) / 2)
    return mid

def basis_fun(p,T,i,t,n): 
    """
    Computes the n'th derivative of basis function i.
    
    Based on algorithm A2.5 from [1]. 
    The algorithm has been modified to include the whole open interval.

    Parameters
    ----------
    p : int
        Order of the spline.
    T : list of floats
        Knot vector.
    i : int
        Basis function number
    t : float
        Variable.
    n : int
        Derivative.

    Returns
    -------
    Float
        Value of B-spline function/derivative at given t
        
    """
    m = len(T)-1
    if (t < T[i] or t >= T[i+p+1]) and (i < m-2*p-1 or t != T[m]): # Local property
        ders = []
        for k in range(n+1): ders.append(0.0)
        return ders[n] # Change if necessary
    N = np.zeros((p+1,p+1)) #initialize N list. Check if correct
    for j in range(p+1):
        N[j][0] = 1.0 if (t >= T[i+j] and t < T[i+j+1]) else 0.0
        if (i+j == m-p-1 and t == T[m]): N[j][0] = 1.0 # Special case
    for k in range(1,p+1):
        saved = 0.0 if (N[0][k-1] == 0.0) else (t-T[i])*N[0][k-1]/(T[i+k]-T[i])
        for j in range(p-k+1):
            Tleft = T[i+j+1]; Tright = T[i+j+k+1]
            if N[j+1][k-1] == 0:
                N[j][k] = saved; saved = 0.0
            else:
                temp = N[j+1][k-1]/(Tright-Tleft)
                N[j][k] = saved + (Tright-t)*temp
                saved = (t-Tleft)*temp
    ders = []; ders.append(N[0][p])
    for k in range(1,n+1):
        ND = N[:,p-k]
        for jj in range(1,k+1):
            saved = 0.0 if (ND[0] == 0.0) else ND[0]/(T[i+p-k+jj]-T[i])
            for j in range(k-jj+1):
                Tleft = T[i+j+1] 
                Tright = T[i+j+p-k+jj+1] # -k inserted. Gives the right results
                if ND[j+1] == 0.0:
                    ND[j] = (p-k+jj)*saved; saved = 0.0
                else:
                    temp = ND[j+1]/(Tright-Tleft)
                    ND[j] = (p-k+jj)*(saved-temp)
                    saved = temp
        ders.append(ND[0])
    return ders[n]# Change if necessary

def spline_calc(x, spf, der = 0):
    """
    Evaluate the smoothing spline at x.

    Parameters
    ----------
    x : float
        x-value to evaluate the smoothing spline at.
    parameters : numpy array
        Vecotr containing the parameters for the smoothing spline.
    p : int
        Order of the smoothing spline.
    t : list
        Knot vector.
    der : int, optional
        Spline derivative. The default is 0.

    Returns
    -------
    y_spline : float
        Spline value at x.

    """
    t, parameters, p = spf
    B = np.zeros(len(t)-p-1)
    for i in range(B.size):
        B[i] = basis_fun(p, t, i, x, der)
    y_spline = np.dot(B, parameters)
    return y_spline