# -*- coding: utf-8 -*-
"""
############################################################################
#  This Python file is used to call bendstiff for calculating the moment-  #
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

This is the main script for calling the functions used in calculating the 
moment-curvature relation of textile materials. The results from the analysis 
is saved in a pickle file in the specified directory. 

Parameters that need to be specified:
    ----------------------------------
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
        
        
Optional parameters (with default values) that may be specified:
    --------------------------------------
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
        
"""
import bendstiff
import pickle
import os
import matplotlib.pyplot as plt

plt.close('all') # close all previous figures upon running script

###### Specify test name and if the parameters should be loaded ##############
test =   'sample_data' # Directory to the test that you want to analyse
test_name ="example_image" # Name of the test you want to analyse
image_name = test_name + ".JPG"

load = False # Should user parameters be loaded or specified

interactive_plot = False # Turn on/off interactive plotting

###### Definition of parameters. Only define if load is set to False #########
# Specimen and image 
W_a = 13.6      # Areal load of fabric N/m^2
w   = 0.05      # Width of fabric specimen m
mpp = 0.1101e-3 # Metres per pixel for the image

# Curve fitting
p = 'auto' # Smoothing spline parameter. If set to 'auto' it will automatically
           # determine the smoothing parameter based in minimisation algorithm.
con = ({'x':0.0, 'f(x)':0.0 , 'der':1 },)  # Constraints on the fit. Use 
                                           # 'x_max' to specify the free end
             
###### Change directory and save/load the parameters #########################
# Change working directory to the directory of the test
cwd  = os.getcwd()
path = os.path.join(cwd,test)
os.chdir(path)
# Make results folder
results_dir = str(test_name)+'_results'
if not os.path.exists(results_dir): 
    os.makedirs(results_dir) 

if load == False:
    par = bendstiff.utility.save_par(test_name, W_a, w, p, mpp, con = con, 
                                     results_dir = results_dir)

#### The following line is an example of how to change the optional parameters
#    par = bendstiff.utility.save_par(test_name, W_a, w, p, mpp, con = con,     \
#                                     smooth = True, plot = True, n_r = 8,      \
#                                     L_kernel = 0.5, n = 50, m = 4, )
else:
    par = bendstiff.utility.load_par(test_name)

# Turn on or off the interactive plotting
plt.ion() if interactive_plot else plt.ioff()
    
###### Run the analysis  #####################################################
x, y, curvature, moment  = bendstiff.run.run_bendstiff(image_name, par)

###### Plot the bending stiffness and save the results #######################
# Plot the bending stiffness
plt.figure('Computed moment-curvature relationship')
plt.plot(curvature, moment)
plt.xlabel('$\kappa$ [m$^{-1} $]')
plt.ylabel('$M$ [N]')
plt.savefig(results_dir+'/resulting_moment_curvature.pdf')

# Save the results
f = open(test_name+'_results.pkl', 'wb')
pickle.dump([x, y, curvature, moment], f)
f.close()

# Write the parameters used to txt file
f = open(results_dir + "/parameters.txt","w")
for key, value in par.items():  
   f.write('%s : %s\n' % (key, value))
f.close()

# set directory to original
os.chdir(cwd)
