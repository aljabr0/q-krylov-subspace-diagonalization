# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Testing Krylov with Re/Im parts of overlaps (small scale, dense).
# This is the first test for the Krylov time reversal method.
# In essence we test that the Krylov method based on the real and imaginary parts
# of respectively the Gram and the Hamiltonian overlaps, converges.
#
# The test compares, for a range of transverse-field strengths gamma, the lowest
# eigenvalue obtained by direct diagonalization of the full Ising Hamiltonian
# against the eigenvalue estimated by the Krylov/QFD procedure below, at two
# different Krylov subspace sizes. The result is a plot of the relative error
# vs. gamma, saved to disk.

import LinearAlgebra: eigen, Hermitian, norm, eigvals, I, diagm
import Plots
include("KTR.jl")
using .KTR

n = 11   # Number of qubits (should be less than 16).
kry_size = 64
h_0, h_d = ising_chain(0, 1, n), ising_chain(1, 0, n) # Base term: -sum(Z); driver term: -sum(XX)
model_name = "ISING"
eigv_idx = 1 # Index of the eigenvalue to track (1 = ground state)

gammas = range(0.05, 2.0, 16) # Sample of parameters

# Starting Krylov vector: |0...0> plus a small uniform admixture over all basis
# states, then normalized. The uniform part ensures overlap with all eigenstates
# so the Krylov subspace doesn't miss components orthogonal to |0...0>.
start_v = [1; zeros(2^n-1)] + ones(2^n) * 2^(-n/2)
start_v = start_v / norm(start_v)

# Estimate the eigv_idx-th eigenvalue of Hamiltonian `h` via the Krylov time
# reversal (QFD-based) method, using a Krylov subspace of size `krylov_size`
# built from `start_v`.
function kry_time_reversal(h, start_v, krylov_size)
    # Check implementation of "krylov_matrix", there is a critical rescaling
    # of the spectrum of H related to the ideal time step.
    # Moreover, the rescaling is applied to the H for the time evolution only.
    # Build the Krylov (QFD) basis: columns are exp(i*t_k*H)*start_v for a
    # sequence of times t_k, with H's spectrum rescaled to fit within [-pi/2, pi/2].
    mat_c = krylov_matrix(h; start_v=start_v, krylov_size=krylov_size, qfd=true,
			    spectrum_scale=pi/2)
    mat_a = Hermitian(im * imag.(mat_c' * h * mat_c)) # Imaginary part of H overlap
    mat_b = Hermitian(real.(mat_c' * mat_c))	# Real part of Gram

    # Spectral thresholding: discard near-null directions of the (possibly
    # ill-conditioned/rank-deficient) Gram matrix mat_b, keeping only the
    # eigenvectors b_u whose eigenvalues b_l exceed the 1e-8 cutoff.
    b_l, b_u = spectral_thr(mat_b, 1e-8)
    # Project mat_a onto the well-conditioned subspace and replace mat_b with
    # the corresponding (now diagonal, invertible) matrix of retained eigenvalues.
    mat_a, mat_b = b_u' * mat_a * b_u, diagm(b_l)
    mat_a, mat_b = map(Hermitian, (mat_a, mat_b))

    # Solve the generalized eigenproblem mat_a * x = l * mat_b * x and return the
    # requested eigenvalue.
    l, u = eigen(mat_a, mat_b)
    return l[eigv_idx]
end

# Reference eigenvalue from direct dense diagonalization, for each gamma.
l0_direct = [eigvals(eval_ham(g, h_0, h_d))[eigv_idx] for g=gammas]
# Krylov estimates of the same eigenvalue, at Krylov size kry_size ...
l0_kry_1 = [kry_time_reversal(eval_ham(g, h_0, h_d), start_v, kry_size) for g=gammas]
# ... and at twice the Krylov size, to check convergence as the subspace grows.
l0_kry_2 = [kry_time_reversal(eval_ham(g, h_0, h_d), start_v, kry_size * 2) for g=gammas]

# Relative error of each Krylov estimate w.r.t. the direct diagonalization result.
l0_krys = [abs.(l0 ./ l0_direct .- 1) for l0=(l0_kry_1, l0_kry_2)]
# Plot relative error vs. gamma for both Krylov sizes, on a log scale by default,
# and save the figure as a PDF.
fig = Plots.plot(gammas, [l0_krys[1] l0_krys[2]], label=[kry_size kry_size *2],
		 linewidth=2,
		 xlabel="γ", ylabel="Rel. Error")
Plots.savefig(fig, "kry-time-r-$model_name.pdf")

