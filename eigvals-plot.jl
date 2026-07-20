# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Plotting utility for the relative error of the ground energy -> time reversal vs QKD

using LaTeXStrings
import Plots: twinx, plot, plot!, savefig
import ITensorMPS: MPS, siteinds, apply, inner
import LinearAlgebra: normalize!, norm, eigvals, I, Hermitian, diagm, eigen, diag, dot
using JLD2
using ArgParse
using Logging

include("KTR.jl")
import .KTR: spectral_thr
include("KTRMPS.jl")
using .KTRMPS

function my_parse_args()
    s = ArgParseSettings()
    @add_arg_table s begin
	"filenames"
	    help = "Input filenames"
	    nargs = '+'
            arg_type = String
	    action = :store_arg
	"--output"
            help = "Output filename"
            arg_type = String
	    default = "plot.png"
	"--ref-energy"
            help = "Ref l0 data filename"
            arg_type = String
	    default = nothing
	"--legend"
            help = "Legend position: best, topright, bottomleft, ..."
            arg_type = String
	    default = "best"
        "--eigvals_thr"
            arg_type = Float32
	    default = 1e-6
	"--xlabel"
            help = "x-axis label"
            arg_type = String
	    default = "x"
	"--ylabel"
            help = "y-axis label"
            arg_type = String
	    default = "\\left|\\lambda_0/\\widehat{\\lambda_0}-1\\right|"
	"--data-labels"
	    nargs = '*'
            arg_type = String
	    action = :store_arg
	"--ref-labels"
	    nargs = '*'
            arg_type = String
	    action = :store_arg
	"--ref-plot-only"
	    help = "Plot only the ref-energy series"
	    action = :store_true
    end

    return parse_args(s)
end

function lambda0(ab; eigvals_thr=1e-6)
    # Solution of the generalized eigenvalue problem Ax=l_0 Bx
    # with spectral thresholding.
    a, b = ab[1:2]
    b_l, b_u = spectral_thr(b, eigvals_thr)
    b = Hermitian(diagm(b_l))
    a = Hermitian(b_u' * a * b_u)

    l, u = eigen(a, b, sortby=x -> real(x))

    if size(ab, 1) > 2
	proj_obs = ab[3]
	# Re-project the projected observable according to the spectral threshold
	proj_obs = Hermitian(b_u' * proj_obs * b_u)
	# TODO Temp inspection just print
	# TODO Verify sign_c
	u0 = u[:, 1]
	@show real(dot(u0, proj_obs, u0))
    end

    return l[1]
end

rel_error_f(v, ref) = abs.((v .- ref) ./ ref)

function process_input(filename, args, l0_ref=nothing)
    my_lambda0 = v -> lambda0(v; eigvals_thr=args["eigvals_thr"])

    @info "Processing $filename"
    jldopen(filename) do f
	params = f["params"]

	l0_tr = map(my_lambda0, f["overlaps_tr"])
	#@warn "l0_tr=$l0_tr"
	l0_qkd  = map(my_lambda0, f["overlaps_qkd"])
	#@warn "l0_qkd=$l0_qkd"
	rel_errors = rel_error_f(l0_tr, l0_qkd)
	rel_errors_ref = nothing
	if !isnothing(l0_ref)
	    rel_errors_ref = rel_error_f(l0_tr, l0_ref)
	end

	label = latexstring("m=$(f["krylov_size"])")

	return rel_errors, rel_errors_ref, params, label
    end
end

function process_ref(filename, args)
    @info "Processing ref energy $filename"
    jldopen(filename) do f
	params = f["params"]
	l0s = f["l0s"]
	#@warn "l0_ref=$l0s"
	return l0s, params
    end
end

function run(args)
    params = nothing
    l0_ref = nothing

    if !isnothing(args["ref-energy"])
	l0_ref, params = process_ref(args["ref-energy"], args)
    end

    rel_errors = []
    rel_errors_ref = []
    labels = []
    for f=args["filenames"]
	rel_error_c, rel_error_cr, params_, label = process_input(f, args, l0_ref)
	params_ = collect(params_)
	push!(rel_errors, rel_error_c)
	!isnothing(rel_error_cr) && push!(rel_errors_ref, rel_error_cr)
	push!(labels, label)
	params = isnothing(params) ? params_ : params
	# Check assumption on same domain for all curves
	@assert all(isapprox.(params .- params_, 0))
    end

    @info "Samples per curve: $(size(params, 1))"
    @info "Axes: $(args["xlabel"]), $(args["ylabel"])"

    if size(args["data-labels"], 1) > 0
	labels = map(latexstring, args["data-labels"])
    end
    ref_labels = labels
    if size(args["ref-labels"], 1) > 0
	ref_labels = map(latexstring, args["ref-labels"])
    end
    labels = reshape(labels, (1, :))
    ref_labels = reshape(ref_labels, (1, :))
    @info "Series labels: $labels"
    @info "Series labels (ref): $ref_labels"

    config = (linewidth=1, linestyle=:dot, markerstrokewidth=0, yscale=:log10)
    plot(minorgrid=true, 
	 legend=Symbol(args["legend"]),
	 xlabel=latexstring(args["xlabel"]), ylabel=latexstring(args["ylabel"]))
    if !args["ref-plot-only"]
	plot!(params, rel_errors; label=labels, marker=:circle, config...)
    end
    if size(rel_errors_ref, 1) > 0
	plot!(params, rel_errors_ref; label=ref_labels, 
	      marker=:utriangle, config...)
    end
    @info "Saving $(args["output"])"
    savefig(args["output"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    run(my_parse_args())
end

