# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Verify the result on the implicit Hadamard test.
#
# Idea: instead of building a Krylov subspace from the "physical" starting
# state |phi> = |+...+> directly (which is what `my_overlaps` below does, and
# is treated here as the ground truth / ordinary "QKD" procedure), we exploit
# the fact that |phi> is not an eigenstate of the time-reversal operator T.
# It decomposes into its T = +1 and T = -1 eigencomponents, v0 and v0_perp.
# Running the Krylov recursion on each eigencomponent separately lets us
# reconstruct the overlap matrices for |phi> using only *half* the number of
# Trotter steps per Krylov vector (since the time-reversal symmetry relates
# the "past" and "future" halves of the sequence) -- this is the "implicit
# Hadamard" trick (see paper). This script builds both the implicit-Hadamard
# overlaps (`my_overlaps_tr`) and the direct overlaps (`my_overlaps`) over a
# range of field strengths `gammas`, and plots their relative error to check
# that the trick reproduces the direct calculation.

using LaTeXStrings
import Plots
import ITensorMPS: MPS, siteinds, apply, inner
import LinearAlgebra: normalize!, norm, eigvals, I, Hermitian, diagm, eigen, diag

include("KTR.jl")
import .KTR: spectral_thr
include("KTRMPS.jl")
using .KTRMPS
using ProgressBars

n = 64			# Chain length (number of spin-1/2 sites)
krylov_size = 128	# Number of Krylov vectors (per starting state) to generate
trotter_config = (cutoff=1e-15, steps_per_unit_t=500, max_bond_dim=500, order=2)
gammas = range(0.01, 0.1, 32) # Sample of transverse-field strengths to test

sites = siteinds("S=1/2", n)

# Physical starting state: all spins pointing along +X.
phi = MPS(Float32, sites, "+")

# Project |phi> onto the T = +1 eigenspace of the time-reversal operator to get
# v0, and onto the T = -1 eigenspace to get v0_perp. Since |phi> = v0 + v0_perp
# (un-normalised), these two pieces are what the implicit-Hadamard trick
# evolves separately.
v0 = apply(ising_chain_time_rev_mpo(sites, 1), phi; cutoff=trotter_config.cutoff)
xi_recip_sq = norm(v0)^2	# 1/xi^2 (see paper)
@show xi_recip_sq
v0_perp = apply(ising_chain_time_rev_mpo(sites, -1), phi; cutoff=trotter_config.cutoff)
PROJ_MIN_NORM = 1e-3
# Guard against a degenerate decomposition (one component ~0 would make the
# corresponding Krylov sequence numerically meaningless).
@assert all([norm(v0), norm(v0_perp)] .>= PROJ_MIN_NORM)
normalize!(v0)
normalize!(v0_perp)

model_name = "Ising"
# Warning, the definitions of H_0 and H_d must match the trotter_ising definitions!
# h_0 is the fixed (XX-coupling) part of the Hamiltonian, h_d is the
# gamma-dependent driving term (transverse field); the full Hamiltonian at a
# given gamma is h_0 + gamma * h_d.
h_0, h_d = ising_chain_mpo(1, 0, sites), ising_chain_mpo(0, 1, sites)
time_r_op = ising_chain_time_rev_mpo(sites)

kry_dt = 0.01
# Time-reversal (implicit Hadamard) Krylov vectors only need half the time
# step per hop, because each overlap effectively combines a forward and a
# backward evolution via T.
mystep_tr(psi::MPS, g::Real) = trotter_ising(g, kry_dt / 2, psi, sites; trotter_config...)
# Direct/explicit Krylov vectors use the full time step, as in ordinary QKD.
mystep_(psi::MPS, g::Real) = trotter_ising(g, kry_dt, psi, sites; trotter_config...)

# Implicit-Hadamard overlap matrices A (Hamiltonian) and B (Gram) for a given
# field strength g, reconstructed from the two T-eigenspace Krylov sequences.
function my_overlaps_tr(g::Real)
    # Krylov sequence generated from the T=+1 component v0.
    kry_vs = krylov_seq(mystep_tr, v0, krylov_size, g)
    a1, b1 = overlaps_time_r_with_sign([h_0, g * h_d], time_r_op, kry_vs)

    # Krylov sequence generated from the T=-1 component v0_perp.
    kry_vs = krylov_seq(mystep_tr, v0_perp, krylov_size, g)
    a2, b2 = overlaps_time_r_with_sign([h_0, g * h_d], time_r_op, kry_vs)

    # Recombine the two eigenspace contributions, weighted by how much of
    # |phi> lies in each (xi_recip_sq and its complement).
    a = a1 * xi_recip_sq + a2 * (1-xi_recip_sq)
    b = b1 * xi_recip_sq + b2 * (1-xi_recip_sq)
    # a, b are built from the (naturally complex, in general non-Hermitian
    # due to numerical noise) raw overlaps; keep only the theoretically
    # expected purely-imaginary part of a and purely-real part of b.
    return im * imag.(a), real.(b)
end

# Overlap procedure for QKD, either: overlap_mps_toe, overlap_mps
const overlap_qkd = overlap_mps_toe

# Direct (ground-truth) overlap matrices A and B, obtained the ordinary QKD
# way: build a single Krylov sequence from |phi> and take overlaps directly,
# without using the time-reversal shortcut.
function my_overlaps(g::Real)
    kry_vs = krylov_seq(mystep_, phi, krylov_size, g)

    b = overlap_qkd(kry_vs)
    a = sum(overlap_qkd(kry_vs, h) for h in [h_0, g * h_d])
    return im * imag(a), real.(b)
end

# Compute both the implicit-Hadamard and the direct overlaps for every
# sampled field strength gamma.
overlaps_tr_per_g = map(my_overlaps_tr, ProgressBar(gammas))
overlaps_per_g = map(my_overlaps, ProgressBar(gammas))

second(v) = v[2]
# Relative Frobenius-norm error between a pair of matrices (implicit-Hadamard
# vs. direct), normalised by the direct ("ground truth") matrix.
rel_error(mats) = norm(mats[1]-mats[2])/norm(mats[2])

# For each of the two matrices (A via `first`, B via `second`), pair up the
# implicit-Hadamard and direct results at each gamma and compute the relative
# error between them.
overlap_rel_error(selector) = map(rel_error, zip(map(selector, overlaps_tr_per_g), map(selector, overlaps_per_g)))
results = map(overlap_rel_error, [first, second])
# Plot the relative error on A and B as a function of gamma: if the implicit
# Hadamard trick is correct, both curves should stay small (numerical noise
# floor) across the sampled range.
fig = Plots.plot(gammas, results, linewidth=1, label=[L"A" L"B"],
		 minorgrid=true, linestyle=:dot,
		 marker=[:diamond :pentagon], markerstrokewidth=0,
                 xlabel=L"\gamma", ylabel=L"\|M - \widehat{M}\|/\|\widehat{M}\|")
Plots.savefig(fig, "time_r-$model_name-impl_hadamard.pdf")

