# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Test ground energy with time reversal vs regular QKD (on Re Gram and Im H overlap).

using LaTeXStrings
import Plots
import ITensorMPS: MPO, MPS, siteinds, apply, inner
import ITensorMPS: Sweeps, setmaxdim!, setcutoff!, dmrg
import LinearAlgebra: normalize!, norm, eigvals, I, Hermitian, diagm, eigen, diag
using JLD2
using ProgressBars

include("KTR.jl")
import .KTR: spectral_thr
include("KTRMPS.jl")
using .KTRMPS

additional_ham_terms = MPO[]
obs = nothing

include("eigvals-recipe_lgt-block.jl")
# include("eigvals-recipe_ising-block.jl")
# include("eigvals-recipe_ising-implicit_hadamard.jl")

omit_qkd = false

_prepare_ham_terms(g::Real) = [[h_0, g * h_d]; additional_ham_terms]

function my_overlaps_tr(g::Real, krylov_size::Int)
    # Overlaps for the Krylov time reversal method.
    # Note we use the states v0 and v0_perp as the initial states here.
    # Such states, must fulfill the required symmetry.
    kry_vs = krylov_seq(mystep_tr, v0, krylov_size, g)
    ham = _prepare_ham_terms(g)
    a1, b1 = overlaps_time_r_with_sign(ham, time_r_op, kry_vs)

    proj_obs1 = 0
    if !isnothing(obs)
	proj_obs1 = overlaps_time_r_with_sign_no_gram(obs, time_r_op, kry_vs)
    end

    if !isnothing(v0_perp)
	kry_vs = krylov_seq(mystep_tr, v0_perp, krylov_size, g)
	a2, b2 = overlaps_time_r_with_sign(ham, time_r_op, kry_vs)

	proj_obs2 = 0
	if !isnothing(obs)
	    proj_obs2 = overlaps_time_r_with_sign_no_gram(obs, time_r_op, kry_vs)
	end

	a = a1 * xi_recip_sq + a2 * (1-xi_recip_sq)
	b = b1 * xi_recip_sq + b2 * (1-xi_recip_sq)
	proj_obs = proj_obs1 * xi_recip_sq + proj_obs2 * (1-xi_recip_sq)
    else
	a, b = a1, b1
	proj_obs = proj_obs1
    end

    return map(Hermitian, (im * imag.(a), real.(b), im * imag.(proj_obs)))
end

# Overlap procedure for QKD, either: overlap_mps_toe, overlap_mps
const overlap_qkd = overlap_mps_toe

function my_overlaps(g::Real, krylov_size::Int; re_im=false)
    # Overlaps for the regular QKD (re_im=false to get the full complex
    # values of the overlaps). Note we use the state phi as the initial state.
    kry_vs = krylov_seq(mystep_, phi, krylov_size, g)

    b = overlap_qkd(kry_vs)
    a = sum(overlap_qkd(kry_vs, h) for h in _prepare_ham_terms(g))
    return re_im ? map(Hermitian, (im * imag.(a), real.(b))) : (a, b)
end

overlaps_tr = [my_overlaps_tr(g, krylov_size) for g=ProgressBar(gammas)]

overlaps_qkd = overlaps_tr
if !omit_qkd	# This option is meant for debugging purposes only
    overlaps_qkd = [my_overlaps(g, krylov_size) for g=ProgressBar(gammas)]
end

jldsave("time_r-overlaps-$model_name-q$n-k$krylov_size.jld2",
	krylov_size=krylov_size, n=n, model_name=model_name,
	overlaps_tr=overlaps_tr, overlaps_qkd=overlaps_qkd,
	params=gammas)

