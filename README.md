# bendstiff

This program is used to call calculate the moment-curvature relation of textiles based on an image of a cantilevered specimen and the areal weight of the textile material. 

<Insert information on the paper when this is out>

The code was developed at Department of Materials and Production at Aalborg University by  P.H. Broberg, E. Lindgaard, C. Krogh, S.M. Jensen, G.G. Trabal, A.F.-M Thai, and B.L.V. Bak. Please feel free to use and adapt the codes but remember to give proper attribution.

Packages needed (version used during development of code)
---------------
- python      (3.8.8)
- numpy       (1.20.1)
- matplotlib  (3.3.4)
- pandas      (1.2.4)
- cv2         (4.0.1)
- scipy       (1.6.2)

Modules
-------
The modules can be found in the /bendstiff directory. They are briefly 
described below.

#### run 
This module contains a function for running the analysis of the 
cantilever bending test.

#### image_processing 
This modules contains functionalities for processing 
the cantilever image and obtaining discrete x and y values for the midline 
deflection. 

#### curvefit 
This module contains functionalities for fitting a smoothing 
spline to a discrete set of data.

#### mocurv 
This module contains functions for calculating the moment and 
curvature of a continuous deflection curve.

#### utility 
This modules contains utility function.

#### bsplines 
This module contains functionalities for calculating the basis 
functions and derivatives.

Input parameters
-------------------------------------------------------------------
### Parameters that need to be specified:

    test : str
        Directory to the image you want to analyse. 
    test_name : str
        Name of the test you want to analyse.
    load : bool
        If load is True, previous parameters (saved in pkl file) is used for 
        the analysis.
    interactive_plot : bool
        If true the generated plots will be showed interactively.     
    W_a : float
        Areal load of the the fabric [N/m^2]
    w : float
        Width of the fabric specimen [m]
    mpp : float
        Metres per pixel conversion factor
    p : float, str
        Smoothing parameter used in the spline fit. Should be a float between 
        0 and 1. If it is specified as 'auto', automatic parameter selection
        is used
    con : dict
        Equality constraints imposed on the spline fit. 
        
        
### Optional parameters (with default values) that may be specified:

    n : int, optional
        Number of data points used in the curve fit. The default is n = 50
    n_r : int, optional
        Number of regions used in the image processing. The default is n_r = 8
    L_kernel : float, optional
        Length of the kernel used in the image processing as a factor of the 
        length of the first region. The default is L_kernel = 0.5
    m : int, optional
        Order of the smoothing spline. The default is m = 4
    plot : bool, optional
        If set to True the script will make plots of each step to help in 
        debugging. The default is plot = False
    smooth : bool, optional
        If set to False the image will not be smoothened during image 
        processing. The default is smooth =  True

How to use
-------------------------------------------------------------------
The program is run through the run module run_bendstiff(image_name, par). 
The main.py script can be used to define the parameters and calling the program.  
It is recommended to run the script with Spyder. This can be installed by downloading the Anaconda distribution. 

Data storage
-------------------------------------------------------------------
The results from the analysis are stored in same folder as the image is located. 
The data is stored in pickle format. 
The parameters used for the analysis is also stored such that they can be loaded again. 

Documentation
-------------------------------------------------------------------
The modules contain docstring and comments describing the functionalities, input
varibale and outputs. 

<The associated journal paper contains a more in-depth 
description of the problem.>

Test
-------------------------------------------------------------------
Type !pytest to test the code.
