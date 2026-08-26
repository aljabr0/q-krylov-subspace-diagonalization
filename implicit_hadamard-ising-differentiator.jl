# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Generate the overlap matrices for numerical demonstration of Lemma F.1.
# For further details on the generation of these matrices, see the
# description in implicit_hadamard-ising.jl

using LaTeXStrings
import Plots
import ITensorMPS: MPS, siteinds, apply, inner
import LinearAlgebra: normalize!, norm, eigvals, I, Hermitian, diagm, eigen, diag

include("KTR.jl")
import .KTR: spectral_thr
include("KTRMPS.jl")
using .KTRMPS
using ProgressBars
using NPZ
using Printf

# Input parameters
length(ARGS) == 8 || error(
    "Expected arguments: --n <integer> --m <integer> " *
    "--r <integer> --k <number>")
ARGS[1] == "--n" || error(
    "Expected --n as the first argument, received: $(ARGS[1])")
ARGS[3] == "--m" || error(
    "Expected --m as the third argument, received: $(ARGS[3])")

ARGS[5] == "--r" || error(
    "Expected --r as the fifth argument, received: $(ARGS[5])")

ARGS[7] == "--k" || error(
    "Expected --k as the seventh argument, received: $(ARGS[7])")

n = parse(Int, ARGS[2]) # Number of qubits
m = parse(Int, ARGS[4]) # Number of Krylov vectors
r = parse(Int, ARGS[6]) # m*r data points for integral/derivative estimators
k = parse(Float64, ARGS[8]) # Trotter steps

krylov_size = m * r  # Total size of A, B
trotter_config = (
    cutoff=1e-15,
    steps_per_unit_t=Int64(500 * k),
    max_bond_dim=500,
    order=2,)

gammas = range(0.01, 0.5, 32)
sites = siteinds("S=1/2", n) # Sites for MPO/MPS definitions

# Physical starting state: all spins pointing along +X (as in the plain
# implicit-Hadamard script). Only used here to compute overlaps/weights and
# as the seed for the direct/ground-truth Krylov sequence in `my_overlaps`.
phi = MPS(Float32, sites, "+")

# Project |phi> onto the T = +1 eigenspace of the time-reversal operator to get
# v0, and onto the T = -1 eigenspace to get v0_perp. Since |phi> = v0 + v0_perp
# (un-normalised), these two pieces are what the implicit-Hadamard trick
# evolves separately.
v0 = apply(ising_chain_time_rev_mpo(sites, 1), phi; cutoff=trotter_config.cutoff)
xi_recip_sq = norm(v0)^2 # 1/xi^2 (see paper)
@show xi_recip_sq
v0_perp = apply(ising_chain_time_rev_mpo(sites, -1), phi; cutoff=trotter_config.cutoff)
PROJ_MIN_NORM = 1e-3

# Guard against a degenerate decomposition
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

kry_dt = 0.1 * (1/r) # Krylov time-step

# Time-reversal (implicit Hadamard) Krylov vectors only need half the time
# step per hop, because each overlap effectively combines a forward and a
# backward evolution via T.
mystep_tr(psi::MPS, g::Real) = trotter_ising(g, kry_dt / 2, psi, sites; trotter_config...)
mystep_(psi::MPS, g::Real) = trotter_ising(g, kry_dt, psi, sites; trotter_config...)

# Implicit-Hadamard overlap matrices A (Hamiltonian) and B (Gram) for a given
# field strength g, reconstructed from the T-eigenspace Krylov sequences generated from v0.
function my_overlaps_tr(g::Real)
    kry_vs = krylov_seq(mystep_tr, v0, krylov_size, g)
    a1, b1 = overlaps_time_r_with_sign([h_0, g * h_d], time_r_op, kry_vs)

    kry_vs = krylov_seq(mystep_tr, v0_perp, krylov_size, g)
    a2, b2 = overlaps_time_r_with_sign([h_0, g * h_d], time_r_op, kry_vs)

    a = a1 * xi_recip_sq + a2 * (1-xi_recip_sq)
    b = b1 * xi_recip_sq + b2 * (1-xi_recip_sq)
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

# Compute both the implicit-Hadamard (blocked) and the direct overlaps for
# every sampled field strength gamma.
overlaps_tr_per_g = map(my_overlaps_tr, ProgressBar(gammas))
overlaps_per_g = map(my_overlaps, ProgressBar(gammas))

# Save data for import by differentiator.py
dict_to_save = Dict{String,Any}(
    "gammas" => gammas,
    "dt"     => kry_dt,
    "m"      => m,
    "r"      => r,
    "n"      => n,
    "k"      => k,
)

# Store direct (ground-truth) overlap matrices A and B
for i in 1:length(overlaps_per_g)
    dict_to_save["a$i"] = overlaps_per_g[i][1]
    dict_to_save["b$i"] = overlaps_per_g[i][2]
end

# Store time-reversal overlap matrices A and B
for i in 1:length(overlaps_tr_per_g)
    dict_to_save["a_tr$i"] = overlaps_tr_per_g[i][1]
    dict_to_save["b_tr$i"] = overlaps_tr_per_g[i][2]
end

# Save both sets of overlap matrices (plus the run parameters) to disk;
# no plot is produced here -- the relative error analysis is performed in differentiator.py
k_filename = @sprintf("%.2f", k)
npzwrite(
    "overlaps_per_g_n$(n)_m$(m)_r$(r)_k$(k_filename).npz",
    dict_to_save,
)