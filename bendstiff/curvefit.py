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

This module contains functionalities for fitting a smoothing spline to a set of
xy-data. 

References
-----------
[1] Piegl, L and Tiller, W. Monographs in Visual Communication, 1997

[2] Wand, MP and Ormerod, JT. On semiparametric regression with O'sullivan 
    penalized splines, Australian and New Zealand Journal of Statistics 
    (50), 2008, 179-198.

[3] Hastie, T, Tibshirani, R and Friedman, J. The Elements of Statistical 
    Learning: Data Mining, Inference, and Prediction. Springer Series in 
    Statistics Vol. 27, 2009
    
[4] Eugene Prilepin and Shamus Husheer, Csaps - cubic spline approximation
    (smoothing).URL - https://github.com/espdev/csaps/pull/47


"""
import bendstiff
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.optimize import minimize_scalar
from bendstiff.bsplines import*

plot_par = [] # Initialize plot parameters list for plotting

def b_matrix(x, t, p):
    """
    Function for calculating the B-matrix. 

    Parameters
    ----------
    x : float
        Parameter.
    t : list
        Knot vector.
    p : int
        Order of the spline.

    Returns
    -------
    B : numpy array
        B-matrix.

    """
    B = np.zeros([x.size, len(t)-p-1])
    for i in range(x.size): 
        span = find_span(x[i], p, t)
        for j in range(span-p,span+1):
            B[i,j] = basis_fun(p, t, j, x[i], 0)
    return B

def omega(t, m):
    """
    Function for calculating the omega-matrix. 
    The numerical integration is based on [2].
    
    Parameters
    ----------
    t : list
        knot vector.
    m : TYPE
        Order of the smoothing spline.

    Returns
    -------
    omega : numpy array
        Omega-matrix.

    """
    x_tilde = np.zeros((2*m-1)*(len(t)-1))
    if m == 1:
        x_tilde = np.array(t)
        wts     = np.repeat(np.diff(t),1) * np.tile((np.array([1])), len(t)-1)
    elif m == 2:
        for i in range(len(t)-1):
            x_tilde[3*i:3*i+3] = np.array([t[i], (t[i]+t[i+1])/2, t[i+1]])
        wts = np.repeat(np.diff(t),3) *                                        \
              np.tile((np.array([1,4,1]))/6, len(t)-1)
    elif m == 3:
        for i in range(len(t)-1):
            x_tilde[5*i:5*i+5] = np.array([t[i], (3*t[i]+t[i+1])/4,            \
                                  (t[i]+t[i+1])/2, (t[i]+3*t[i+1])/4, t[i+1]])
        wts = np.repeat(np.diff(t),5) *                                        \
              np.tile((np.array([14,64,8*3,64,14]))/(45*4), len(t)-1)
    elif m == 4:
        for i in range(len(t)-1):
            x_tilde[7*i:7*i+7] = np.array([t[i], (5*t[i]+t[i+1])/6,            \
                                 (2*t[i]+t[i+1])/3, (t[i]+t[i+1])/2,           \
                                 (t[i]+2*t[i+1])/3, (t[i]+5*t[i+1])/6, t[i+1]])
        wts = np.repeat(np.diff(t),7) *                                        \
              np.tile((np.array([41,216,27,272,27,216,41]))/(140*6), len(t)-1)
    else:
        print('Invalid order of smoothing spline. m should be between 1 and 4')
    
    Bdd = np.zeros([(2*m-1)*(len(t)-1), len(t)-2*m])
    
    for i in range(Bdd.shape[0]): # Make this banded at some point
        for j in range(Bdd.shape[1]):
            Bdd[i,j] = basis_fun((2*m-1), t, j, x_tilde[i], m)
    omega = np.transpose(Bdd) @ np.diag(wts) @ Bdd
    return omega

def add_constraints(M, b, t, p, constraints):
    """
    Function for adding Lagrange multipliers to the linear equations. 

    Parameters
    ----------
    M : numpy array
        Matrix for linear equation.
    b : numpy vecotr
        Vector for linear equation.
    t : list
        Knot vector.
    p : int
        Order of the spline.
    constraints : list of dictionaries
        Contains the equality constraints to be added to the equations.

    Returns
    -------
    A : numpy array
        Updated matrix for linear equation.
    bb : numpy vector
        Updated vector for the linear equation.

    """
    if len(constraints) == 0:
        return M,b
    R = sp.lil_matrix((len(constraints), M.shape[0]))
    c = np.zeros(len(constraints))
    for i in range(len(constraints)):
        span = find_span(constraints[i]['x'], p, t)
        # Calculate R matrix
        for j in range(span-p,span+1):
            R[i,j] = basis_fun(p,t,j,constraints[i]['x'], constraints[i]['der'])
        # Calculate c matrix 
        c[i] = constraints[i]['f(x)']
    
    zero = sp.lil_matrix((R.shape[0], R.shape[0]))
    A1   = sp.hstack([M, np.transpose(R)], format = 'csr')
    A2   = sp.hstack([R, zero], format = 'csr')
    A    = sp.vstack([A1, A2])
    bb   = np.hstack((b,c))
    return A, bb

def cross_validation(lam, y, B, omega_jk):
    """
    Calculates the CV value for a given smoothing parameter lam, using leave-
    one-out cross validation. Based on Eq. (5.26) and (5.27) in [3]

    Parameters
    ----------
    lam : float
        Non-normalised smoothing parameter.
    y : numpy vector
        y-values
    B : numpy array
        B-matrix.
    omega_jk : nump array
        Omega-matrix.

    Returns
    -------
    float
        CV value at the given lam.

    """
    smoother = B @ np.linalg.inv(np.transpose(B) @ B +                         \
                                 lam * omega_jk) @ np.transpose(B)  
    f_lam = smoother @ y
    cv = 0.0
    for i in range(y.size):
        cv += ((y[i]-f_lam[i])/(1-smoother[i,i]))**2 
    return 1/y.size*cv 

def smoothing_spline(x, y, spar, m = 2, thin=False, con = (), segments = 0):
    """
    Evaluating the smoothing spline for a xy-dataset. The output are spline
    parameters that are compatible with the SciPy's spline evaluations. 

    Parameters
    ----------
    x : numpy array
        Vector of x-values.
    y : numpy array
        Vector of y-values.
    spar : float
        Smoothing parameter.
    m : int, optional
        Order of the smoothing spline. The default is 2.
    thin : bool, optional
        If knot thinning should be used. The default is False.
    con : list of dics, optional
        Constraints for the Lagrange multipliers. The default is ().
    segments : int, optional
        Segments used for the knots. The default is 0.

    Returns
    -------
    spf : tuple
        Spline parameters. Compatible with SciPy's splines (can be read by 
        splev)
    B   : numpy array
        B-matrix.
    omega_jk : numpy array
        Omega-matrix.

    """
    p = 2*m-1 # Spline degree
    t = knots(list(x),p, thin, segments) # Change to numpy array at a point

    #### Create b-matrix ####
    B = b_matrix(x, t, p)
    Bt = np.transpose(B)

    #### Calculate Omega matrix ####
    omega_jk = omega(t, m)
    
    #### Calculate spar (if auto) ######
    
    if spar == 'auto':
        print('Automatic smoothing parameter selection started:')
        spar = auto_smooth(x, y, B, omega_jk, m)
        print('Automatic smoothing parameter is '+str(spar))
    
    if spar == 1.0: #lam == 0.0:
        if B.shape[0] > B.shape[1]: # Inserted 21 march 2022
            M = Bt @ B
            M = sp.lil_matrix(M)
            b = np.matmul(Bt, y)
            A,bb = add_constraints(M, b, t, p, con)
            parameters = la.spsolve(A,bb)
            parameters = parameters if len(con) == 0                           \
                                    else parameters[:-len(con)]
            spf = (np.array(t), np.array(parameters), p) 
            return spf, B, omega_jk
        # Solve the underdetermined set of equations using least norm
        A_dagger = Bt @ np.linalg.inv(B @ Bt)
        parameters = A_dagger @ y
        spf = (np.array(t), np.array(parameters), p) 
        return spf, B, omega_jk, spar
    
    #### Calculate Lambda ####
    spar_norm, k = normalize_smooth(x, spar, m)
    lam = k*(1-spar)/spar
   
    ### Add weights NEW ####
    w = np.logspace(1, 1, x.size)
    w = w / (np.sum(w)/w.size)   
    W = np.diag(w)
    
    #### Add constraints #####
    M = Bt @ W @ B + lam*omega_jk # w inserted
    M = sp.lil_matrix(M)
    b = np.matmul(Bt, y)
    A,bb = add_constraints(M, b, t, p, con)

    #### Solve linear equation ####
    # Using standard solver
    parameters = la.spsolve(A,bb)
    parameters = parameters if len(con) == 0 else parameters[:-len(con)]
    spf = (np.array(t), np.array(parameters), p) 
    return spf, B, omega_jk, spar

def normalize_smooth(x, smooth, m):
    """
    Normalise the smoothing parameter using a modified approach of [4]

    Parameters
    ----------
    x : numpy array
        Vector of x-values.
    smooth : float
        Non-normalised smoothing parameter.
    m : int
        Order of the smoothing spline.

    Returns
    -------
    p : float
        Normalised smoothing parameter.
    k : TYPE
        DESCRIPTION.

    """
    span    = np.ptp(x)
    factor  = 2*m-1
    w       = 1/x.size * np.ones(x.size)
    eff_x   = 1 + (span ** 2) / np.sum(np.diff(x) ** 2)
    eff_w   = np.sum(w) ** 2 / np.sum(w ** 2)
    k       = factor**m * (span ** factor) * (x.size ** (-2*(factor/3))) *     \
              (eff_x ** -(0.5*(factor/3)))  * (eff_w ** -(0.5*(factor/3))) 
    p       = smooth / (smooth + (1 - smooth) * k)
    return p, k

def cv_fun(p, x, y, B, omega_jk, sp_order):
    spar, k = normalize_smooth(x, p, sp_order)
    lam_cv = k*(1-p)/p
    CV = cross_validation(lam_cv, y, B, omega_jk)
    return CV

def auto_smooth(x, y, B, omega_jk, sp_order):
    """
    Function to automatically chose the smoothing parameter based on 
    minimisation of the scalar CV function.

    Parameters
    ----------
    x : array
        x values of the deflection curve.
    y : array
        y values of the deflection curve.
    B : array
        B-matrix from the smoothing spline.
    omega_jk : array
        Omega matrix from the smoothing spline.
    sp_order : int
        order of the smoothing spline.

    Returns
    -------
    spar : float
        Automatically chosen smoothing parameter.

    """
    res = minimize_scalar(cv_fun, args=(x,y,B,omega_jk, sp_order),             \
                          bounds =(0,1), method='bounded')
    spar = res.x
    return spar

def cv_plot(x, y, B, omega_jk, par, spar_plot):
    """
    Function for plotting the CV values

    Parameters
    ----------
    x : array
        x values of the deflection curve.
    y : array
        y values of the deflection curve.
    B : array
        B matrix from the smoothing spline
    omega_jk : array
        Omega matrix from the smoothing spline.
    par : dict
        parameters used for the analysis.
    spar_plot : float
        Smoothing parameter used for the analysis.

    """
    n_cv = 100 
    p_cv = np.linspace(0.01, 0.99999,n_cv) 
    lam_cv = np.zeros(n_cv)
    for i in range(n_cv):
        spar, k = normalize_smooth(x, p_cv[i], par['m'])
        lam_cv[i] = k*(1-p_cv[i])/p_cv[i]
        cv = np.zeros(n_cv)
    for i in range(n_cv):
        cv[i] = cross_validation(lam_cv[i], y, B, omega_jk)
    plt.figure('CV PLOT')
    plt.plot(p_cv,cv, label='CV values')
    plt.axvline(spar_plot, color='r', label='Chosen CV value')
    plt.title('CV')
    plt.xlabel('p'); plt.ylabel('CV')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    return
    

def run_curvefit(x, y, par):
    """
    Main function for running the smoothing spline curve fit

    Parameters
    ----------
    x : numpy array
        x-values.
    y : numpy array
        y-values.
    par : dict
        Parameters used.

    Returns
    -------
    spf : tuple
        Parameters for the smoothing spline.

    """
    # Carry out smoothing spline
    spf, B, omega_jk, spar = smoothing_spline(x, y, par['p'], m=par['m'],      \
                             thin = False, con = par['con'])
    
    cv_plot(x, y, B, omega_jk, par, spar) # Plot the CV value
    
    if par['plot'] == True:
        
        fig,axes = bendstiff.utility.subplot_creator('Curvefit',2,3)
        
        x_sp = np.linspace(0 , np.max(x), 100)
        y_sp = np.array([spline_calc(a, spf) for a in x_sp]) 
        plot_par.append( {"Title": 'Data points', "Order": 0,                  \
                          "Type": 'scatter', "Data": (x,y)} )
        plot_par.append( {"Title": 'Spline', "Order": 1, "Type": 'xy',         \
                          "Data": (x_sp,y_sp)} )
        
        # Spline derivative. Plots to used to check if constraints are satisfid
        y_spd = np.array([spline_calc(a, spf, der=1) for a in x_sp]) 
        plot_par.append( {"Title": 'Spline derivative', "Order": 2,            \
                          "Type": 'xy', "Data": (x_sp,y_spd)} )
        
        # Spline curvature. Check of constraints
        sec_der = np.array([spline_calc(a, spf, der = 2) for a in x_sp]) 
        fir_der = np.array([spline_calc(a, spf, der = 1) for a in x_sp]) 
        y_spc   = - sec_der / np.power((1+fir_der**2), 1.5)
        
        plot_par.append( {"Title": 'Spline curvature' , "Order": 3,            \
                          "Type": 'xy', "Data": (x_sp,y_spc)} )
        
        # Plot centerline after erosion
        plot_par.append( {"Title": 'Fabric deflection (Post erosion)',         \
                          "Order": 4, "Type": 'scatter', "Data": (x,y),        \
                          'Label':'Data points'} )
        plot_par.append( {"Title": 'Fabric deflection (Post erosion)',         \
                          "Order": 4, "Type": 'xy', "Data": (x_sp,y_sp),       \
                          'Label': 'Spline'} )
        
        # Fill out the subplots
        bendstiff.utility.subplot_filler(axes,plot_par)
    return spf