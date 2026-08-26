# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This script employs the derivative estimation scheme proposed in
# https://arxiv.org/abs/2412.17289 , and the standard cumulative simpson's 
# integration method in order to produce the plots in Figure 8 of the manuscript.
#
# The workflow is:
# 1. Load exact and densely sampled overlap matrices, generating either file
#    with the Julia script when it is not already present.
# 2. Estimate the Hamiltonian overlap matrix A by differentiating samples of B.
# 3. Estimate the Gram overlap matrix B by cumulatively integrating samples of A.
# 4. Apply spectral thresholding, solve the generalized eigenvalue problems,
#    and compare the indirect estimates with the sampled matrices.
# 5. Save the relative eigenvalue-error and matrix-error plots as PDF files.


import numpy as np
from numpy import load
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson,cumulative_trapezoid,cumulative_simpson
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
from pathlib import Path
import subprocess

# Fonts for publication figures.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

# Regularized estimator for a scalar function and its first two derivatives.
# The estimator combines the initial Taylor polynomial defined by ``xin`` with
# a kernel correction fitted to measurements ``y`` at sample times ``ts``.
class DerivativeEstimator:
    def __init__(self, y, ts, xin, thetas):
        """
        y: np.array of (noisy) measurement values of the exact function z.

        ts: timepoints corresponding to measurement values.

        xin: initial vector with components xin_{k} = d^{k}/dt^{k} z(t)|_{t=0}.

        thetas: list of regularisation parameters [theta1,theta2]. 
        See https://arxiv.org/abs/2412.17289 for details.
        """
        self.y = y
        self.ts = ts
        self.xin = xin
        self.M = 3
        D = len(ts)

        if thetas is not None:
            self.theta1 = thetas[0]
            self.theta2 = thetas[1]
        else: 
            # put eta, f norm estimator here
            raise NotImplementedError
        
        c = np.array([self.x0(ts[i])[0] for i in range(D)])
        P = np.zeros((D,D),dtype=complex)
        for ja in range(D):
            for ka in range(D):
                P[ja,ka] = self.xs(ts[ja],ts[ka])[0]

        self.P = P
        self.c = c
        self.alpha = la.solve((1/self.theta2)*np.eye(D) + P, y - c)


    # Evaluate the initial quadratic trajectory and its derivatives.
    def x0(self,t):
        [k1,k2,k3] = self.xin
        return np.array([(k3*(t**2))/2 + k2*t + k1,k3*t + k2,k3])

    # Evaluate the correction kernel and its first two derivatives.
    def xs(self,t,ts_i):

        def Heaviside(t):
            if t < 0: return 0
            if t >= 0: return 1
            
        xs_0 = lambda t,ts_i: (t**3*(t**2 - 5*t*ts_i + 10*ts_i**2) - (t - ts_i)**5*Heaviside(t - ts_i)) / (120*self.theta1)
        xs_1 = lambda t,ts_i: (t**2*(t**2 - 4*t*ts_i + 6*ts_i**2) - (t - ts_i)**4*Heaviside(t - ts_i)) / (24*self.theta1)
        xs_2 = lambda t,ts_i: (t*(t**2 -3*t*ts_i + 3*ts_i**2) - (t - ts_i)**3*Heaviside(t - ts_i)) / (6*self.theta1)
        
        return np.array([xs_0(t,ts_i),xs_1(t,ts_i),xs_2(t,ts_i)])

    # Evaluate the fitted function and its first two derivatives.
    def eval(self,t):
        assert 0<=t<=np.max(self.ts), "evaluation point outside domain of time measurements"
        return self.x0(t) + sum([self.alpha[i]*self.xs(t,self.ts[i]) for i in range(len(self.ts))])


# Return the Hermitian transpose of a matrix.
def dag(A): return np.conjugate(A.T)

# Project a generalized eigenvalue problem onto the eigenspace of Sm whose
# eigenvalues exceed ``thres``, removing numerically unstable directions.
def threshold_eig(Hm,Sm,thres=1e-10):
    valS,vecS = la.eig(Sm)
    idxS = valS.argsort()[::-1] 
    valS = valS[idxS]
    vecS = vecS[:,idxS]
    cutoff_ind = np.where(np.real(valS) > thres)[0][-1]
    valS_trunc = valS[:cutoff_ind+1]
    vecS_trunc = vecS[:,:cutoff_ind+1]
    H_trunc = dag(vecS_trunc) @ Hm @ vecS_trunc 
    S_trunc = dag(vecS_trunc) @ Sm @ vecS_trunc
    return H_trunc,S_trunc

# Apply common grid, axes, font, and legend formatting to a figure.
def format_plot(ylabel):
    ax = plt.gca()
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="both", linewidth=0.5, color="0.95")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r"$\gamma$", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(fontsize=18)

# Run data generation, reconstruction, error analysis, and plotting only when
# this file is executed as a script.
if __name__ == "__main__":

    # Configure the system size, Krylov dimension, and dense sampling ratio.
    n = 48 # system size
    m = 10 # krylov dimension
    r = 20 # number of datapoints for integral/derivative estimators

    # Read in data from implicit_hadamard-ising-differentiator

    # Load the standard-resolution reference overlaps. If the archive is
    # absent, generate it with the Julia script using r=1 and k=k_exact.
    k_exact = 0.4 # Trotter steps
    exact_path = f"overlaps_per_g_n{int(n)}_m{int(m)}_r1_k{k_exact:.2f}.npz"
    if not Path(exact_path).exists():
        subprocess.run(
            ["julia",
            "implicit_hadamard-ising-differentiator.jl",
            "--n", str(n),
            "--m", str(m),
            "--r", str(1),
            "--k", str(k_exact),
            ],
            check=True)
    data_exact = dict(np.load(exact_path))

    # Load densely sampled overlaps for differentiation and integration. If
    # absent, generate them with the Julia script using the configured r and k.
    k = 5 # Trotter steps
    der_int_path = f"overlaps_per_g_n{int(n)}_m{int(m)}_r{int(r)}_k{k:.2f}.npz"
    if not Path(der_int_path).exists():
        subprocess.run(
            ["julia",
            "implicit_hadamard-ising-differentiator.jl",
            "--n", str(n),
            "--m", str(m),
            "--r", str(r),
            "--k", str(k),
            ],
            check=True,)
    data = dict(np.load(der_int_path))

    # Recover the simulation parameters stored with the dense overlap data.
    dt, gammas = data["dt"], data["gammas"]
    M = len(gammas)

    a_err,b_err,a_err_mst,b_err_mst = [],[],[],[]
    for i in range(1,M+1):
        a = data_exact["a"+str(i)]
        b = data_exact["b"+str(i)]
        a_tr = data_exact["a_tr"+str(i)]
        b_tr = data_exact["b_tr"+str(i)]
        a_msmt = data["a_tr"+str(i)]
        b_msmt = data["b_tr"+str(i)]

        a_err.append(la.norm(a - a_tr)/la.norm(a))
        b_err.append(la.norm(b - b_tr)/la.norm(b))

        a_err_mst.append(la.norm(a - a_msmt[::r,::r])/la.norm(a))
        b_err_mst.append(la.norm(b - b_msmt[::r,::r])/la.norm(b))

    # Reconstruct A from a derivative estimate of B, and reconstruct B from a
    # cumulative integral estimate of A.

    a_tr_minimax_gamma,b_tr_minimax_gamma = [],[]
    a_gamma,b_gamma = [],[]
    a_err,b_err = [],[]
    a_err_minimax,b_err_minimax = [],[]
    b_from_int_err = []
    a_err_msmt,b_err_msmt = [],[]
    lambda0,lambda0_tr_msmt,lambda0_tr_minimax,lambda0_tr_integral = [],[],[],[]
    a_err_tr_subsample,b_err_tr_subsample = [],[]


    for i in range(1,M+1):

        # Use regularly spaced dense samples as the comparison matrices.
        a = data["a_tr"+str(i)][::r,::r]
        b = data["b_tr"+str(i)][::r,::r]

        a_tr_subsample = data["a_tr"+str(i)][::r,::r]
        b_tr_subsample = data["b_tr"+str(i)][::r,::r]

        a_tr = data_exact["a_tr"+str(i)]
        b_tr = data_exact["b_tr"+str(i)]

        a_tr_msmt = data["a_tr"+str(i)]
        b_tr_msmt = data["b_tr"+str(i)]

        y = data["b_tr"+str(i)][0,:]
        D = len(y)
        assert abs(D - r*m)==0
        assert all( matrix.shape == (m, m) for matrix in (a, b, a_tr, b_tr) )

        ts = np.arange(D)
        cs_exact = CubicSpline(np.arange(D),y)

        k1 = 1
        k2 = 0
        k3 = cs_exact(ts,2)[0]
        xin = [1,0,k3]

        theta1 = 1e-15
        theta2 = 1

        x_hat = DerivativeEstimator(y,ts,xin,[theta1,theta2])

        # evaluated estimates
        times = ts[::r] # back to original coords
        x0 = np.array([x_hat.eval(ti)[0] for ti in times])
        x1 = np.array([x_hat.eval(ti)[1] for ti in times])
        x2 = np.array([x_hat.eval(ti)[2] for ti in times])

        assert len(x0) == len(x1) == a.shape[0]

        # Assemble Hermitian Toeplitz matrices from the fitted estimates.
        b_tr_minimax = np.triu(la.toeplitz(x0)) + dag(np.triu(la.toeplitz(x0),1))
        a_tr_minimax = 1j*np.triu(la.toeplitz(1/(dt)*x1)) - 1j*dag(np.triu(la.toeplitz(1/(dt)*x1),1))

        a_tr_minimax_gamma.append(a_tr_minimax)
        b_tr_minimax_gamma.append(b_tr_minimax)

        a_gamma.append(a)
        b_gamma.append(b)

        a_err.append(la.norm(a - a_tr,ord='fro')/la.norm(a,ord='fro'))
        b_err.append(la.norm(b - b_tr,ord='fro')/la.norm(b,ord='fro'))

        a_err_minimax.append(la.norm(a - a_tr_minimax,ord='fro')/la.norm(a,ord='fro'))
        b_err_minimax.append(la.norm(b - b_tr_minimax,ord='fro')/la.norm(b,ord='fro'))

        # Recover the first row of B by cumulative Simpson integration, then
        # use Hermitian Toeplitz structure to assemble the complete matrix.
        input = np.imag(data["a_tr"+str(i)][0,:])
        b_from_integral_row1 = (2*(2.5e-3)*cumulative_simpson(input,initial=0) + 1)[::r]
        b_from_integral = np.triu(la.toeplitz(b_from_integral_row1)) + dag(np.triu(la.toeplitz(b_from_integral_row1),1))
        b_from_int_err.append(la.norm(b - b_from_integral,ord='fro')/la.norm(b,ord='fro'))


        a_err_tr_subsample.append(la.norm(a_tr_subsample - a_tr_minimax,ord='fro')/la.norm(a_tr_subsample,ord='fro'))
        b_err_tr_subsample.append(la.norm(b_tr_subsample - b_from_integral,ord='fro')/la.norm(b_tr_subsample,ord='fro'))

        a_err_msmt.append(la.norm(a - a_tr_msmt[::r,::r],ord='fro')/la.norm(a,ord='fro'))
        b_err_msmt.append(la.norm(b - b_tr_msmt[::r,::r],ord='fro')/la.norm(b,ord='fro'))

        # Threshold each overlap pair before solving the generalized
        # eigenvalue problem and retaining its smallest eigenvalue.
        thres = 5e-7
        a_thres,b_thres = threshold_eig(a,b,thres=thres)
        a_tr_msmt_thres,b_tr_msmt_thres = threshold_eig(a_tr_msmt[::r,::r],b_tr_msmt[::r,::r],thres=thres)
        a_tr_minimax_thres,b_tr_minimax_thres = threshold_eig(a_tr_minimax,b_tr_msmt[::r,::r],thres=thres)
        a_tr_integral_thres,b_tr_integral_thres = threshold_eig(a_tr_msmt[::r,::r],b_from_integral,thres=thres)

        lambda0.append(np.min(la.eig(a_thres,b_thres)[0]))
        lambda0_tr_msmt.append(np.min(la.eig(a_tr_msmt_thres,b_tr_msmt_thres)[0]))
        lambda0_tr_minimax.append(np.min(la.eig(a_tr_minimax_thres,b_tr_minimax_thres)[0]))
        lambda0_tr_integral.append(np.min(la.eig(a_tr_integral_thres,b_tr_integral_thres)[0]))

    # Plot relative errors in the lowest generalized eigenvalue.
    plt.figure(figsize=(12, 8))
    plt.semilogy(
        gammas,
        np.abs(np.array(lambda0_tr_msmt) - np.array(lambda0_tr_minimax))
        / np.abs(np.array(lambda0_tr_msmt)),
        "^",
        linestyle=(0, (2, 2)),
        markersize=10,
        color="dodgerblue",
        label=r"$(A_{\text{derivative}},B)$",
    )
    plt.semilogy(
        gammas,
        np.abs(np.array(lambda0_tr_msmt) - np.array(lambda0_tr_integral))
        / np.abs(np.array(lambda0_tr_msmt)),
        "^",
        linestyle=(0, (2, 2)),
        markersize=10,
        color="coral",
        label=r"$(A,B_{\text{integral}})$",
    )
    format_plot(r"$\left|\lambda_0/\widehat{\lambda}_0 - 1\right|$")
    plt.savefig("time_r-Ising-indirect-energy-estimates.pdf", dpi=300)
    plt.show()

    # Plot relative Frobenius-norm errors in the reconstructed matrices.
    plt.figure(figsize=(12, 8))
    plt.semilogy(
        gammas,
        a_err_minimax,
        "^",
        linestyle=(0, (2, 2)),
        markersize=10,
        color="dodgerblue",
        label=r"$A_{\text{derivative}}$",
    )
    plt.semilogy(
        gammas,
        b_from_int_err,
        "^",
        linestyle=(0, (2, 2)),
        markersize=10,
        color="coral",
        label=r"$B_{\text{integral}}$",
    )
    format_plot(r"$|M - \widehat{M}|/|\widehat{M}|$")
    plt.savefig("time_r-Ising-indirect-matrix-estimates.pdf", dpi=300)
    plt.show()