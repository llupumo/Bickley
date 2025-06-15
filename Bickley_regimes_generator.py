#!/usr/bin/env python
# coding: utf-8

# We start by adding the necessary folders to the current working path.

# import sys/os
import sys, os
import matplotlib.pyplot as plt
import numpy as np

# get current directory
path = os.getcwd()

# get parent directory
parent_directory = "/home/llu/Desktop/TBarrier-main/TBarrier/2D"

# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/utils")

# add integration folder to current working path
sys.path.append(parent_directory+"/subfunctions/integration")

# add HyperbolicsLCS folder to current working path
sys.path.append(parent_directory+"/demos/AdvectiveBarriers/HyperbolicLCS")



# Import scipy
import scipy.io

# Import numpy
import numpy as np


# In[63]:


def k(n, Len_X):
    
    return 2*n*np.pi/Len_X

def psi0(params, y):
    
    U = params[0]
    Len_Y = params[3]
    c2 = params[7]
    
    return -U*Len_Y*np.tanh(y/Len_Y)+c2*y

def psi1(params, x, y, t):
    
    U = params[0]
    R = params[1]
    Len_X = params[2]
    Len_Y = params[3]
    eps1 = params[4][0]
    eps2 = params[4][1]
    A = params[5]
    c1 = params[6]
    c2 = params[7]

    sigma1 = k(1, Len_X)*(c1-c2)
    
    forcing2 = np.cos(k(2, Len_X)*x)
    forcing1 = np.cos(k(1, Len_X)*x-sigma1*t)
    
    return A*U*Len_Y*(sech(y/Len_Y)**2)*(eps1*forcing1 + eps2*forcing2)

def sech(x):
    
    return 2*np.exp(x)/(np.exp(2*x)+1)

def tanh(x):
    
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def psi(params, x, y, t):

    return psi0(params, y)+psi1(params, x, y, t)

def Bickley_jet(params, x, y, t, dx, dy):
    
    X, Y, time = np.meshgrid(x, y, t)
    
    u = np.zeros(X.shape)
    v = np.zeros(Y.shape)
    
    for i in range(X.shape[0]):
        
        for j in range(Y.shape[1]):
            
            for r in range(time.shape[2]):
                
                x_ = X[i, j, r]
                y_ = Y[i, j, r]
                t_ = time[i, j, r]
                
                u[i, j, r] = -(psi(params, x_, y_ + dy, t_)-psi(params, x_, y_- dy, t_))/(2*dy)
                v[i, j, r] = (psi(params, x_ + dx, y_, t_)-psi(params, x_ - dx, y_, t_))/(2*dx)
    
    return u, v




# U = 62.66*3600*24 (in kilometers/day)
u = 62.66*3.6*24

random_A = np.random.uniform(0, 1 , 5)
random_c2divu = np.random.uniform(0, 1 , 5)*1.6
pairs = zip(random_A, random_c2divu)
# Print the pairs
for A, q_c2 in pairs:
    c2 = q_c2*u #0.461*u
    #q_c1 = (c2/u-(0.461-0.205))  #
    q_c1 = 0.205
    c1=q_c1*u
    formatted_A = f"{A:.8f}"
    formatted_c2divu = f"{q_c2:.8f}"
    formatted_c1divu = f"{q_c1:.8f}"

    # Radius of the Earth (in kms)
    earthRadius = 6371

    # L_x = pi*earthRadius
    Len_X = np.pi*earthRadius

    # L_y = 1.77*10**3 (in km)
    Len_Y = 1.77*(10**3)

    # [epsilon_1, epsilon_2]
    epsilon = [.15, .3]


    sigma1 = k(1, Len_X)*(c1-c2)

    T_periodic = abs(2*np.pi/(sigma1))

    # time array
    time_data = np.linspace(0, T_periodic, 20, endpoint = True)
    x = np.arange(-1.2*Len_X, 1.2*Len_X, 200)
    y = np.arange(-3*Len_Y*1.25, 3*Len_Y*1.25, 200)

    params = [u, earthRadius, Len_X, Len_Y, epsilon, A, c1, c2]

    # Define grid-spacing for spatial derivative
    dx = Len_X/5000
    dy = Len_Y/5000

    U, V = Bickley_jet(params, x, y, time_data, dx, dy)


                
    scipy.io.savemat("/home/llu/Desktop/LCS_graph/Bickley/flow/Bickley_A_"+formatted_A+"_cdivU_"+formatted_c2divu+"_c1divU_"+formatted_c1divu+".mat", {'u': U, 'v': V, 'x': x, 'y': y, 't': time_data, 'A' : A, 'c2divu' : q_c2, 'c1divu' : q_c1 })

    x.reshape(1,x.shape[0])
    y.reshape(1,y.shape[0])
    time_data.reshape(1,time_data.shape[0])


    time_data = time_data.reshape(1,time_data.shape[0])
    x = x.reshape(1,x.shape[0])
    y = y.reshape(1,y.shape[0])

    # Print the shapes of the variables
    print("Shape of u:", U.shape)
    print("Shape of v:", V.shape)
    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)
    print("Shape of t:", time_data.shape)
    # Example: Access specific data
    print("First few values of x:", x[0,:5])


    # # Computational parameters and data
    # 
    # Here we define the computational parameters and the data.

    # In[67]:


    # import numpy
    import numpy as np

    # Number of cores for parallel computing
    Ncores = 5 # int

    # Time resolution of data
    dt_data = time_data[0, 1]-time_data[0,0] # float

    # Periodic boundary conditions
    periodic_x = False # bool
    periodic_y = False # bool
    periodic_t = True # bool
    periodic = [periodic_x, periodic_y, periodic_t]

    # Unsteady velocity field
    bool_unsteady = True # bool

    # Defined domain
    defined_domain = np.isfinite(U[:,:,0]).astype(int) # array (NY, NX)

    ## Compute meshgrid of dataset
    X, Y = np.meshgrid(x, y) # array (NY, NX), array (NY, NX)

    ## Resolution of meshgrid
    dx_data = X[0,1]-X[0,0] # float
    dy_data = Y[1,0]-Y[0,0] # float

    delta = [dx_data, dy_data] # list (2, )


    # # Spatio-temporal domain
    # 
    # Here we define the spatio-temporal domain over which to consider the dynamical system.

    # In[68]:


    # Initial time (in days)
    t0 = 0 # float

    # Final time (in days)
    tN = time_data[0,-1] # float

    # Time step-size (in days)
    dt = .1 # float

    # NOTE: For computing the backward trajectories set: tN < t0 and dt < 0.

    time = np.arange(t0, tN+dt, dt) # shape (Nt,)

    # Length of time interval (in days)
    lenT = abs(tN-t0) # float

    # Longitudinal and latitudinal boundaries (in degrees)
    xmin = x.min() # float
    xmax = x.max() # float
    ymin = y.min() # float
    ymax = y.max() # float

    # Make sure that the chosen domain is part of the data domain
    assert np.min(X) <= xmin <= np.max(X), " xmin must be between "+f'{np.min(X)} and {np.max(X)}'
    assert np.min(X) <= xmax <= np.max(X), " xmax must be between "+f'{np.min(X)} and {np.max(X)}'
    assert np.min(Y) <= ymin <= np.max(Y), " ymin must be between "+f'{np.min(Y)} and {np.max(Y)}'
    assert np.min(Y) <= ymax <= np.max(Y), " ymax must be between "+f'{np.min(Y)} and {np.max(Y)}'
    assert np.min(time_data) <= t0 <= np.max(time_data), " t0 must be between "+f'{np.min(time_data)} and {np.max(time_data)}'
    assert np.min(time_data) <= tN <= np.max(time_data), " tN must be between "+f'{np.min(time_data)} and {np.max(time_data)}'

    # Spacing of meshgrid (in degrees)
    dx = 230 # float
    dy = 50 # float

    x_domain = np.arange(xmin, xmax + dx, dx) # array (Nx, )
    y_domain = np.arange(ymin, ymax + dy, dy) # array (Ny, )

    X_domain, Y_domain = np.meshgrid(x_domain, y_domain) # array (Ny, Nx)

    # Define ratio of auxiliary grid spacing vs original grid_spacing
    aux_grid_ratio = .2 # float between [1/100, 1/5]
    aux_grid = [aux_grid_ratio*(X_domain[0, 1]-X_domain[0, 0]), aux_grid_ratio*(Y_domain[1, 0]-Y_domain[0, 0])] # list (2, )


    # # Interpolate velocity
    # 
    # In order to evaluate the velocity field at arbitrary locations and times, we interpolate the discrete velocity data. The interpolation with respect to time is always linear. The interpolation with respect to space can be chosen to be "cubic" or "linear". Defulat value is "cubic".

    # In[69]:


    # Import interpolation function for unsteady flow field
    from ipynb.fs.defs.Interpolant import interpolant_unsteady

    # Set nan values to zero (in case there are any) so that we can apply interpolant. 
    # Interpolant does not work if the array contains nan values. 
    U[np.isnan(U)] = 0
    V[np.isnan(V)] = 0

    # Interpolate velocity data using cubic spatial interpolation
    Interpolant = interpolant_unsteady(X, Y, U, V, method = "cubic")

    Interpolant_u = Interpolant[0] # RectangularBivariateSpline-object
    Interpolant_v = Interpolant[1] # RectangularBivariateSpline-object


    # # Hyperbolic LCS from forward computation

    # ## Cauchy Green (CG) strain tensor
    # 
    # The Cauchy Green strain tensor $ C_{t_0}^t(\mathbf{x}) $ is computed by using an auxiliary meshgrid.

    # In[70]:


    # Import function to compute gradient of flow map
    from ipynb.fs.defs.gradient_flowmap import gradient_flowmap

    # Import function to compute Cauchy-Green strain tensor
    from ipynb.fs.defs.CauchyGreen import CauchyGreen

    # Import package for parallel computing
    from joblib import Parallel, delayed

    # Split x0, y0 into 'Ncores' equal batches for parallel computing
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def parallel_C(x0_batch, y0_batch):
        
        # Initial conditions
        X0 = np.array([x0_batch, y0_batch]) # array (2, Nx*Ny)

        # Compute gradient of flow map
        gradFmap = gradient_flowmap(time, X0, X, Y, Interpolant_u, Interpolant_v, periodic, defined_domain, bool_unsteady, time_data, aux_grid) # shape (N, Nx*Ny, 2, 2)

        # extract gradient from t0 to tN
        gradFmap_t0_tN = gradFmap[-1,:, :, :] # array (N, 2, 2, Nx*Ny)

        # compute CG
        C = gradFmap_t0_tN.copy()*np.nan
        for i in range(gradFmap_t0_tN.shape[2]):
            C[:,:,i] = CauchyGreen(gradFmap_t0_tN[:,:,i]) # array (2, 2, Nx*Ny)
        
        return C


    # In[71]:


    # Vectorize initial conditions by arranging them to a vector of size (Nx*Ny, )
    x0 = X_domain.ravel() # array (Nx*Ny,)
    y0 = Y_domain.ravel() # array (Nx*Ny,)

    x0_batch = list(split(x0, Ncores)) # list (Nx*Ny,)
    y0_batch = list(split(y0, Ncores)) # list (Nx*Ny,)

    results = Parallel(n_jobs=Ncores, verbose = 50)(delayed(parallel_C)(x0_batch[i], y0_batch[i]) for i in range(len(x0_batch)))

    # extract CG tensorfield from results of parallel computing
    C = results[0]

    for res in results[1:]:
        C = np.append(C, res, axis = 2)
        
    # reshape array from vectorized form to structured meshgrid
    C = C.reshape((2, 2, X_domain.shape[0], X_domain.shape[1])) # array (2, 2, Ny, Nx)


    # ## Eigenvalues/Eigenvectors of Cauchy Green strain tensor
    # 
    # We now compute the eigenvalues and eigenvectors of Cauchy Green strain tensor. We additionally also compute the $ \mathrm{FTLE}_{t_0}^{t_N} $ as we seek to later compare the repelling LCS with features of the FTLE-field.

    # In[72]:


    # Import function to compute eigenvalues/eigenvectors
    from ipynb.fs.defs.eigen import eigen

    # Import function to check location of particle
    from ipynb.fs.defs.check_location import check_location

    # add FTLE folder to current working path
    sys.path.append(parent_directory+"/demos/AdvectiveBarriers/FTLE2D")

    # Import function to calculate FTLE from Cauchy-Green strain tensor
    from ipynb.fs.defs.FTLE import _FTLE_C

    # maximum eigenvalues of CG-tensor
    eig1 = C[0,0,:,:].copy()*np.nan # array (Ny, Nx)
    # minimum eigenvalues of CG-tensor
    eig2 = C[0,0,:,:].copy()*np.nan # array (Ny, Nx)
    # eigenvectors associated to maximum eigenvalues of CG-tensor
    e1 = np.zeros((eig1.shape[0], eig1.shape[1], 2))*np.nan # array (Ny, Nx, 2)
    # eigenvectors associated to minimum eigenvalues of CG-tensor
    e2 = np.zeros((eig2.shape[0], eig2.shape[1], 2))*np.nan # array (Ny, Nx, 2)
    # FTLE-field
    FTLE = C[0,0,:,:].copy()*np.nan # array (Ny, Nx)

    #iterate over meshgrid
    for i in range(X_domain.shape[0]):
        
        for j in range(Y_domain.shape[1]):
                
            x = [X_domain[i,j], Y_domain[i, j]]
            
            # only compute CG tensor for trajectories starting region where velocity field is defined
            if check_location(X, Y, defined_domain, np.array(x))[0] == "IN":
            
                # compute eigenvalues and eigenvectors of CG tensor
                eig1[i,j], eig2[i,j], e1[i,j,:], e2[i,j,:] = eigen(C[:,:,i,j])
                
                # compute FTLE-field from CG tensor
                FTLE[i, j] = _FTLE_C(C[:,:,i,j], lenT)


    # Define figure/axes
    fig = plt.figure(figsize=(8, 4), dpi=600)  # Set figure size and resolution
    ax = plt.axes()
    # Add FTLE field to plot
    cax = ax.contourf(X_domain, Y_domain, np.ma.masked_invalid(FTLE), levels=400, cmap="gist_gray")
    # Set axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Set axis ticks
    ax.set_xticks(np.arange(xmin, xmax + 0.1, 20000))
    ax.set_yticks(np.arange(ymin, ymax + 0.1, 2000))
    # Set axis labels
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    # Colorbar
    cbar = fig.colorbar(cax, ticks=np.arange(0, 1, 0.1))
    cbar.ax.set_ylabel(r'FTLE', rotation=0, fontsize=6)
    # Title
    ax.set_title("Forward FTLE: A: "+formatted_A+"  c/U: "+formatted_c2divu+"  c1/U: "+formatted_c1divu, fontsize=10)
    # Save the figure with consistent size and no extra whitespace
    plt.savefig(
        "/home/llu/Desktop/LCS_graph/Bickley/flow/Bickley_A_"+formatted_A+"_c2divU_"+formatted_c2divu+"_c1divU_"+formatted_c1divu+".png",
        bbox_inches='tight',  # Ensures no extra whitespace
        pad_inches=0.1        # Adds a small padding around the figure
    )
