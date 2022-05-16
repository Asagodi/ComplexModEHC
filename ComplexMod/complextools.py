import os, sys
sys.path.append(os.getcwd())
sys.path.append("C:/Users/abel_/Documents/Rotations/IoR/Code/FlexModEHC/")

import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

from generators import Generator
from propagators import Propagator

from utils import norm_density, ensure_dir
from scipy.stats import sem

from scipy.optimize import LinearConstraint, Bounds
from scipy.optimize import NonlinearConstraint

tol = 10**-3

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedir(dirname)

def eigendecompose(gen_O):
    "returns (modulated) eigenspectrum and generator eigendecomposition"
    n_states = gen_O.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(gen_O)
    W = eigenvectors
    G = np.linalg.inv(W)
    return eigenvalues, G, W

def calcul_Gamma(G, W):
    n_states = G.shape[0]
    Gamma = np.zeros((n_states,n_states,n_states), dtype = 'complex_')
    for i in range(n_states):
        for j in range(n_states):
            for k in range(n_states):
                Gamma[k,i,j] = G[i,k]*W[k,j]
    return Gamma

def zcf_gen(s, W, deltaT, rho):
    """ FUNCTION: computes generated autocorrelation
    INPUTS: s       = spectrum
            W       = spectral components (n_k,n_s,n_s)
            deltaT  = lags to compute
            rho     = state probability vector
    """
    deltaT = np.asarray(deltaT)
    n_s = rho.size
    n_t = deltaT.size
    #not necessary to add: dtype = 'complex_'
    Wd = np.array([W[:,i,i] for i in range(n_s)]).T # (n_s,n_k)
    ZCgen = np.array([s]* n_t).T # (n_k,n_t) convention
    ZCgen = ZCgen**deltaT
    ZCgen = Wd@ZCgen # (n_s,n_k) x (n_k,n_t) -> (n_s,n_t)
    ZCgen = rho@ZCgen # (,n_s) x (n_s,n_t) -> (n_t,)
    return ZCgen


def circle_diffusion(n_state, noise=False, symmetric=False):
    """Creates a generator (gen_O) for a diffusion process on a circle"""
    gen_O = np.zeros((n_state,n_state))
    if noise != False:
        k, epsilon = noise
    else:
        k, epsilon = 1, 0.
    gen_O_row = np.zeros(n_state)
    gen_O_row[0] = -(1+epsilon)
    gen_O_row[1] = 1.
    gen_O_row[2:k+2] = epsilon/k
    if symmetric == True:
        gen_O_row[-k:] = epsilon/k
        gen_O_row[0] -= epsilon
    for i in range(n_state):
        gen_O[i,:] = np.roll(gen_O_row, i)
    return gen_O


def rotate(z, phi, c=1):
    "rotating eigenvalues by $\varphi$"
    #should extract center? #     center = 1 + epsilon or c=1+2epsilon #does not work for noisy diffusion
    return (z + c)*np.exp(phi*1j)-c

def calculate_etO(power_spec, V, Vinv, tau=1, alpha=1, t=1, sigma=1):
    """power_spec = PROP.GEN.evals_fwd
        V = PROP.GEN.EVEC_fwd
        Vinv = PROP.GEN.EVECinv_fwd"""
    n_state = power_spec.shape[0]
    sigma_alpha = sigma ** (2 * alpha)
    power_spec_alpha = power_spec ** (alpha)
    x = sigma_alpha * power_spec_alpha * t / tau
    d = np.exp(x)
    
    etD = (np.eye(n_state) * d)  # d = power_spec, etD=spectral power as a diagonal matrix
    spec_comp_weights = np.ones((n_state,))
    wetD = spec_comp_weights * etD
    spec_basis = np.matmul(V, wetD)
    etO = np.matmul(spec_basis, Vinv)
    return etO


def circle_diffusion(n_state, noise=False, symmetric=False):
    """Creates a generator (gen_O) for a diffusion process on a circle"""
    gen_O = np.zeros((n_state,n_state))
    if noise != False:
        k, epsilon = noise
    else:
        k, epsilon = 1, 0.
    gen_O_row = np.zeros(n_state)
    gen_O_row[0] = -(1+epsilon)
    gen_O_row[1] = 1.
    gen_O_row[2:k+2] = epsilon/k
    if symmetric == True:
        gen_O_row[-k:] = epsilon/k
        gen_O_row[0] -= epsilon
    for i in range(n_state):
        gen_O[i,:] = np.roll(gen_O_row, i)
    return gen_O


def simulate(etO, T, rho0):
    rho_t = rho0
    rhos = [rho_t]
    for t in range(T):
        rho_t = np.dot(rho_t, etO)
        rhos.append(rho_t)
        
    return np.array(rhos)

def sample_complex(etO, rho_start_idx, n_samp= 1000, n_step = 15):
    # simulation with negative and complex spectrum
    lags_plot = np.arange(0,n_step)
    n_state = etO.shape[0]
    state_seqs = np.zeros((n_samp, n_step))
    rhos = np.zeros((n_samp, n_step, n_state))
    rho_start = process_rho(rho_start_idx, n_state)
    for ns in range(n_samp):

        state_seqs[ns,0] = rho_start_idx
        rho_state = rho_start
        rho_state = rho_state/rho_state.sum()
        rhos[ns,0,:] = rho_state
        p = rho_state.real+rho_state.imag #p = rho_state.real #p = rho_state
        p_sign = np.sign(p)
        p = norm_density(np.abs(p), beta=1.0, type='l1')
        state = np.random.choice(list(range(len(p))), 1, p=np.abs(p)) #     state = np.random.choice(list(range(len(p))), 1, p=p)
        rho_inter = process_rho(state, n_state) # sampled state as prior for next step
        rho_inter *= p_sign
        for n in range(1,n_step):
            rho_stop = evolve_complex(etO, n_step=1, rho_start=rho_inter)
            rho_state = rho_stop/rho_stop.sum()
            p = rho_state.real+rho_state.imag #p = rho_state.real
            p_sign = np.sign(p)
            p = np.abs(p)/np.sum(np.abs(p)) #norm_density(p, beta=1.0, type='l1')
            state = np.random.choice(list(range(len(p))), 1, p=p) 
            state = p_sign[state[0]]*state
            #store
            state_seqs[ns,n] = state
            rhos[ns,n,:] = rho_state

            #calculate rho for next step
            rho_inter = process_rho(state, n_state)
            rho_inter *= p_sign
    return state_seqs, rhos

def sign_0(x):
    y = 1
    if x<0.:
        y=-1
    return y

def sample_stats(data, d=0):
    #calculate statistics on simulations
    # instead of #(x==y), we need x\dot y:
    n_samp = data.shape[0]
    n_t = data.shape[1]

    AC_samp = np.zeros((n_samp, n_t))
    for k in range(n_t):
        AC_samp[:,k] = (data[:,0]==np.abs(data[:,k]))
    #     AC_samp[k,:] = (np.abs(start_prop-np.abs(data[k,:]))<=d)
#         AC_samp[k,:] = (np.abs(np.abs(data[0,:])-np.abs(data[k,:]))<=d) #<=d doesnt work for TJunction or OpenBox
        for i in range(n_samp):
            AC_samp[i,k] *= sign_0(data[i,k])

    AC = AC_samp.mean(0)
    AC_sem = sem(AC_samp, axis=0)
    return AC, AC_sem
    
def process_rho(rho, n_state):
    """
    FUNCTION: Process state distribution.
    INPUTS: rho = state distribution or state
    OUTPUTS: rho_out = state distribution
    NOTES: rho = None returns a uniform distribution
           rho = state returns a one-hot distribution
           else rho_out==rho
    """
    if rho is None:
        rho_out = np.ones((n_state))
        return rho_out/rho_out.sum()
    elif not hasattr(rho, "__len__") or np.array(rho).size < n_state:
        rho_out = np.zeros((n_state))
        rho_out[np.asarray(rho).astype('int')] = 1.
    else:
        rho_out = rho
    return rho_out/rho_out.sum()

def evolve_complex(etO, n_step=1, rho_start=None):
    """
    Evolves prior state density rho_start forward to rho_stop using self.PROP
    INPUTS: rho_start = prior state density to evolve
            n_step = number of steps to evolve
            ignore_imag = False, raises error
    NOTES: N/A
    """
    for i in range(n_step):
        rho_start = np.dot(rho_start, etO)
#         rho_start = norm_density(rho_start, beta=1.0, type='l1') # L1 normalize
    return rho_start


def characteristic_fun(k, mu, c, alpha):
    return np.exp(1j*k*mu-np.abs(c*k)**alpha)


def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z))) #if z.shape=0 then cannot concatenate

def complex_abs(z):
    return np.sqrt(np.real(z)**2 + np.imag(z)**2) #make scalar: |z|=sqrt(a+bi)

def linear_sum(z):
    return np.real(z) + np.imag(z) #make scalar: f(z)=a+b


def constraints_stochmat_ri(W, tol=tol):
    """
    FUNCTION: linear constraints which ensure resulting evolution matrix is a stochastic matrix and Im(W[k,i,j]*s_k) = 0
    INPUTS: W = (2, n_k, n_s, n_s) 2 copies of spectral weights
    #imaginary constraint: # sum_{j,k} W[k,i,j]*b_k = 0 for s_k=(a_k+ib_k)
    """
    n_k = W.shape[0]
    Wreal = np.zeros((2*n_k, 2*n_k, 2*n_k))
    Wreal[:n_k, :n_k, :n_k] = W
    WrealT = np.zeros((2*n_k, 2*n_k, 2*n_k))
    WrealT[:n_k, :n_k, :n_k] = W.T
    B = np.zeros(((2*n_k)**2, 2*n_k))
    B[:n_k**2, :n_k] = W.T.reshape(-1,n_k)
    Wimag = np.zeros((2*n_k, 2*n_k, 2*n_k))
    Wimag[-n_k:, -n_k:, -n_k:] = W
    
    # sum_{j,k} W[k,i,j]*a_k = 1
    lb = np.array([(1-tol)*np.ones(n_k), -tol*np.ones(n_k)]).reshape(-1)
    ub = np.array([(1+tol)*np.ones(n_k), tol*np.ones(n_k)]).reshape(-1)
    lc1_stochmat = LinearConstraint(A=WrealT.sum(0), lb=lb.reshape(-1), ub=ub.reshape(-1)) #lb <= A.dot(x) <= ub
    
    # sum_{k} W[k,i,j]*a_k >= 0
    lb = -tol*np.ones((2*n_k, 2*n_k))
    ub = np.inf*np.ones((2*n_k, 2*n_k))
    ub[:n_k, :n_k] = tol*np.ones((n_k, n_k))
    lc2_stochmat = LinearConstraint(A=B, lb=lb.reshape(-1), ub=ub.reshape(-1)) #will reshape work correctly?
    
    # sum_{k} W[k,i,j]*b_k = 0 
    lb = -tol*np.ones((2*n_k, 2*n_k))
    ub = tol*np.ones((2*n_k, 2*n_k))
    lc3_stochmat = LinearConstraint(A=Wimag.T.reshape((-1, 2*n_k)), lb=lb.reshape(-1), ub=ub.reshape(-1))
    
    # sum_{j,k} W[k,i,j]*b_k = 0  
    lb = np.array([-tol*np.ones(n_k), tol*np.ones(n_k)]).reshape(-1)
    ub = np.array([-tol*np.ones(n_k), tol*np.ones(n_k)]).reshape(-1)
    lc4_stochmat = LinearConstraint(A=Wimag.T.sum(0), lb=lb.reshape(-1), ub=ub.reshape(-1))
    
    return (lc1_stochmat, lc2_stochmat, lc3_stochmat, lc4_stochmat)


# lc1_stochmat, lc2_stochmat, lc3_stochmat, lc4_stochmat = constraints_stochmat_ri(W)
#constraints_stochmat should be rewritten to have Wreal=(W[0],0)

def real_imag_Gamma(W):
#     W\in \mathbb{R}^{nxnxn}
#     bigW\in \mathbb{R}^{2nx2nx2n} with W in diagonal
    n_k = W.shape[0]
    bigW = np.zeros((2*n_k, 2*n_k, 2*n_k))
    bigW[:n_k, :n_k, :n_k] = W
    bigW[-n_k:, -n_k:, -n_k:] = W
    return bigW


def bound_by_norm(n_state, tol=tol):
    n_k = n_state  #int(s.shape[0]/2)
    con = lambda x: [np.sqrt(x[i]**2+x[i+n_k]**2) for i in range(n_k)] # [np.norm()] #[|s_1|, |s_2|, ..., |s_n|]
    lb = -tol*np.ones(n_k)
    ub = (1+tol)*np.ones(n_k)
    return NonlinearConstraint(fun=con, lb=lb, ub=ub)

def conjugate_constraint(n_state, tol=tol):
    n_k = n_state
    A = np.zeros((2*n_k, 2*n_k))
    A[n_k+1:, n_k+1:] = np.diag(np.ones(n_k-1))
    b = np.diag(np.ones((1, n_k-2))[0], 1)
    A[n_k+1:, n_k+1:] -= b
    A[n_k, n_k] = 1.
    return LinearConstraint(A=A, lb=-tol, ub=tol)


def plot_fullprop(PROP, ax, start, cbar=True):
    kix = start
    first_state = kix
    cmap_grid_code = plt.cm.jet
    vmin = (PROP.etO[first_state : first_state + 1, :].real).flatten().min()
    vmax = (PROP.etO[first_state : first_state + 1, :].real+PROP.etO[first_state : first_state + 1, :].imag).flatten().max()
    PROP.ENV.plot_state_func(
                    state_vals=PROP.etO[kix, :].real+PROP.etO[kix, :].imag,
                    ax=ax,
                    cbar=cbar,
                    cmap=plt.cm.RdBu_r,
                    vmin=vmin,
                    vmax=vmax,
                    mask_color="white",
                    norm=None,
                    arrows=False,
                )