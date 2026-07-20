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
import ITensorMPS: MPS, MPO, siteinds, apply, inner, OpSum
import ITensorMPS: Sweeps, setmaxdim!, setcutoff!, dmrg
import LinearAlgebra: normalize!, norm, eigvals, I, Hermitian, diagm, eigen, diag
using JLD2
using ProgressBars

include("KTR.jl")
import .KTR: spectral_thr
include("KTRMPS.jl")
using .KTRMPS

additional_ham_terms = MPO[]

# Recipes, include just the selected one
include("eigvals-recipe_lgt-block.jl")
#include("eigvals-recipe_ising-block.jl")

_prepare_ham_terms(g::Real) = [[h_0, g * h_d]; additional_ham_terms]

l0s = Float32[]
if !@isdefined sweeps
    global sweeps = Sweeps(10)
    setmaxdim!(sweeps, 10, 20, 100, 200, 500)
    setcutoff!(sweeps, 1e-15)
end

for g=gammas
    @show "**** **** **** ****"
    ham = _prepare_ham_terms(g)
    l0, psi0 = dmrg(ham, phi, sweeps)
    push!(l0s, l0)

    if @isdefined dmrg_gs_check
	dmrg_gs_check(psi0)
    end
end

@show extrema(l0s)

jldsave("time_r-overlaps-$model_name-q$n-dmrg.jld2",
	n=n, model_name=model_name, l0s=l0s,
	params=gammas)

