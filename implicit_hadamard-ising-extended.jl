# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Generate the overlaps for the extended (blocked) implicit Hadamard test.
#
# Idea: `implicit_hadamard-ising.jl` splits the physical state |phi> =
# |+...+> into just two time-reversal eigencomponents (T = +1 and T = -1).
# This script generalises that trick: |phi> is instead decomposed (in the
# GHZ/domain-wall basis built by `mps_ghz_blocks`) into `2^v0_block_count`
# components v0s, each of which is *also* a T-eigenstate (with eigenvalue
# `signs[i]`), by cutting the chain into `v0_block_count` blocks and fixing
# a definite (+ or -) relative sign pattern across blocks. Running the
# Krylov recursion separately on each of these components and recombining
# them (weighted by their overlap with |phi>) reconstructs the overlap
# matrices for |phi> using half the Trotter steps per Krylov vector, exactly
# as in the plain implicit-Hadamard trick, but now with finer-grained
# eigencomponents. `proj_terms` lets you use only the first `proj_terms`
# (of the `2^v0_block_count` possible) components, to test how the
# reconstruction accuracy degrades as fewer terms are kept. Rather than
# plotting the relative error directly (as the non-extended script does),
# this script just computes both sets of overlap matrices and saves them to
# disk (via JLD2) for later, offline analysis.

using LaTeXStrings
import Plots
import ITensorMPS: MPS, siteinds, apply, inner
import LinearAlgebra: normalize!, norm, eigvals, I, Hermitian, diagm, eigen, diag, tr

include("KTR.jl")
import .KTR: spectral_thr
include("KTRMPS.jl")
using .KTRMPS
using JLD2
using ProgressBars

n = 64			# Number of qubits
krylov_size = 128	# Number of Krylov vectors
v0_block_count = 4	# GHZ block count, must be a divisor of n and the resulting blocks have size multiple of 4
proj_terms = 2		# Between 1 and 2^v0_block_count
trotter_config = (cutoff=1e-15, steps_per_unit_t=500, max_bond_dim=500, order=2)	# Trotter config
gammas = range(0.01, 0.1, 32) # Sample of parameters for the Hamiltonian

sites = siteinds("S=1/2", n) # Sites for MPO/MPS definitions

# Physical starting state: all spins pointing along +X (as in the plain
# implicit-Hadamard script). Only used here to compute overlaps/weights and
# as the seed for the direct/ground-truth Krylov sequence in `my_overlaps`.
phi = MPS(Float32, sites, "+")
# Enumerate all 2^v0_block_count sign patterns c ∈ {-1,+1}^v0_block_count,
# one per block: `digits(i, base=2, pad=v0_block_count)` gives the bits of i,
# and `2*bit-1` maps 0/1 to -1/+1.
v0s = [2*digits(i, base=2, pad=v0_block_count).-1 for i=(0:(2^v0_block_count - 1))]
# Each v0s[i] is (up to numerical noise) a T-eigenstate with eigenvalue equal
# to the product of its block signs (verified below).
signs = prod.(v0s)
# Build the actual GHZ-block MPS for each sign pattern: block "++" if the
# corresponding sign is -1, "-+" if it is +1 (see `mps_ghz_blocks`). These
# 2^v0_block_count states play the role that v0/v0_perp play in the plain
# implicit-Hadamard script, but there are more of them and each is supported
# on a finer decomposition of the chain into blocks.
v0s = [mps_ghz_blocks(c, sites, "++", "-+") for c in v0s]
# Weight of each component in the decomposition of |phi>, i.e. how much of
# |phi> lies along each v0s[i] (the analogue of xi_recip_sq/1-xi_recip_sq in
# the plain script, generalised to 2^v0_block_count components).
xis_recip_sq = [abs(inner(phi, v0))^2 for v0 in v0s]
@show sum(xis_recip_sq)

model_name = "Ising"
# Warning, the definitions of H_0 and H_d must match the trotter_ising definitions!
# h_0 is the fixed (XX-coupling) part of the Hamiltonian, h_d is the
# gamma-dependent driving term (transverse field); the full Hamiltonian at a
# given gamma is h_0 + gamma * h_d.
h_0, h_d = ising_chain_mpo(1, 0, sites), ising_chain_mpo(0, 1, sites)
time_r_op = ising_chain_time_rev_mpo(sites)

# Verify signs: confirm each v0s[i] really is an eigenstate of the
# time-reversal operator T with the eigenvalue predicted from its block sign
# pattern (`signs[i]`), up to numerical noise.
@assert all(isapprox.([real(inner(v0', time_r_op, v0)) for v0=v0s] .- signs, 0; atol=1e-5))

kry_dt = 0.01
# Time-reversal (implicit Hadamard) Krylov vectors only need half the time
# step per hop, because each overlap effectively combines a forward and a
# backward evolution via T.
mystep_tr(psi::MPS, g::Real) = trotter_ising(g, kry_dt / 2, psi, sites; trotter_config...)
# Direct/explicit Krylov vectors use the full time step, as in ordinary QKD.
mystep_(psi::MPS, g::Real) = trotter_ising(g, kry_dt, psi, sites; trotter_config...)

# Implicit-Hadamard overlap matrices A (Hamiltonian) and B (Gram) for a given
# field strength g, reconstructed from the first `terms` (out of
# 2^v0_block_count) T-eigenspace Krylov sequences generated from v0s.
function my_overlaps_tr(g::Real, terms=proj_terms)
    as, bs = [], []
    for i=1:terms
	# Krylov sequence generated from the i-th T-eigenspace component v0s[i].
	v0 = v0s[i]
	kry_vs = krylov_seq(mystep_tr, v0, krylov_size, g)
	a, b = overlaps_time_r_with_sign([h_0, g * h_d], time_r_op, kry_vs)
	push!(as, a)
	push!(bs, b)
    end

    # Recombine the `terms` eigenspace contributions, weighted by how much of
    # |phi> lies in each (xis_recip_sq).
    weights = xis_recip_sq[1:terms]
    # No need to re-normalize the weights as such (positive) rescale cancels out
    # in the generalized eigenvalue problem
    #weights = weights ./ sum(weights)

    a_sum = sum(a * c for (a, c)=zip(as, weights))
    b_sum = sum(b * c for (b, c)=zip(bs, weights))
    # a_sum, b_sum are built from the (naturally complex, in general
    # non-Hermitian due to numerical noise) raw overlaps; keep only the
    # theoretically expected purely-imaginary part of a_sum and purely-real
    # part of b_sum, then force exact Hermitian symmetry.
    return map(Hermitian, (im * imag.(a_sum), real.(b_sum)))
end

# Overlap procedure for QKD, either: overlap_mps_toe, overlap_mps
const overlap_qkd = overlap_mps_toe

# Direct (ground-truth) overlap matrices A and B, obtained the ordinary QKD
# way: build a single Krylov sequence from |phi> and take overlaps directly,
# without using the time-reversal shortcut. If `re_im` is set, apply the same
# real/imaginary-part projection and Hermitian symmetrisation as
# `my_overlaps_tr`, so the two can be compared directly.
function my_overlaps(g::Real, re_im=false)
    kry_vs = krylov_seq(mystep_, phi, krylov_size, g)

    b = overlap_qkd(kry_vs)
    a = sum(overlap_qkd(kry_vs, h) for h in [h_0, g * h_d])
    return re_im ? map(Hermitian, (im * imag.(a), real.(b))) : (a, b)
end

# Compute both the implicit-Hadamard (blocked) and the direct overlaps for
# every sampled field strength gamma.
overlaps_tr_per_g = map(my_overlaps_tr, ProgressBar(gammas))
overlaps_per_g = map(my_overlaps, ProgressBar(gammas))

# Save both sets of overlap matrices (plus the run parameters) to disk;
# unlike the non-extended script, no plot is produced here -- the relative
# error analysis is done offline from this saved data.
jldsave("time_r-overlaps-$model_name-q$n-k$krylov_size-implicit_hadamard_ext-projte$proj_terms.jld2",
	krylov_size=krylov_size, n=n, model_name=model_name,
	overlaps_tr=overlaps_tr_per_g, overlaps_qkd=overlaps_per_g,
	params=gammas)

