# PyTAMS

This directory contains Python 3 scripts implementing the Trajectory Adaptive Multilevel Sampling algorithm (TAMS), a variant of Adaptive Multilevel Splitting (AMS), for the study of rare events.

The `demo` folder contains the necessary files to run the TAMS algorithm. 

The `main.py` file is the file to be executed using a command of the type `python main.py`. In `main.py`, the parameters for the TAMS algorithm are specified (trajectory time, time step, score function, number of particles, type of score threshold, maximum number of iterations, noise level etc.). It calls the class `TAMS_object` defined in `TAMS_class_nd.py` where the core of the TAMS algorithm is written. `main.py` also called the dynamical system file (`triple_well.py` in the demo) in which the system parameters (drift, diffusion matrix, target and initial state etc.) are specified. 

The time stepping scheme is handled in `schemes.py`. It currently uses the Euler-Maruyama scheme. To improve performance, the time stepping could be wrapped from lower level languages.

The demo system consists in a 2D double well gradient system with a potential wall. Plotting tools for the demo system are given in `tools_2D.py`. Score functions (valid in any dimensions) are also given in `triple_well.py`. 

Confidence ellipsoids, used for setting target sets, are computed in the `ellipsoid_fun.py` file. The `warp_score_function_ell.py` file incorporates the confidence ellipsoid as target set into the score function.

Output files are written to a `.hdf5` output file.

The main directory contains files for the analysis of the performance of TAMS and its results, (they are not essential for running TAMS) such as a series of files which estimate the typical transition path from the spatial histogram of transition trajectories and a method to design a score function based on the estimated transition path. The directory also includes dynamical system files for the Lorenz 1963 model and a 4-mode model of the double-gyre wind-driven ocean circulation.


