# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Recipe for eigvals.jl, do not invoke this directly.
# Case LGT with blocked initial state.

n = 64			# Number of qubits
v0_block_count = 2	# GHZ block count, must be a divisor of n and the resulting blocks have size multiple of 4
krylov_size = 80
trotter_config = (cutoff=1e-15, steps_per_unit_t=500, max_bond_dim=500, order=2)
gammas = range(1., 1.2, 32) # Sample of parameters

model_name = "LGTZ2-ghzb$v0_block_count"

sites = siteinds("S=1/2", n)
gauge_sector = 1	# Either 1 or -1 (sector -1 still under construction)
@assert gauge_sector in (1, -1)

function dmrg_gs_check(psi0)
    # Check gauge invariant ground state and selection of the right sector
    @assert isapprox(inner(psi0', lgtz2_gauge_mpo(sites, gauge_sector), psi0), 0; atol=1e-5)
    @assert isapprox(inner(psi0', lgtz2_gauge_mpo(sites, -gauge_sector), psi0), 1; atol=1e-5)
    @show inner(psi0', lgtz2_gauge_mpo(sites, 0), psi0)

    # Print overlap of the various initial states with the ground state
    @show inner(psi0', phi)
    @show inner(psi0', v0)
end

# This state is gauge invariant for this Hamiltonian
# TODO Initial state for gauge_sector<0 is a placeholder...
phi = MPS(Float32, sites, gauge_sector>0 ? "+" : "-")
# However, after the projection induced by the operator T, the new state loses the property
v0 = mps_ghz_blocks(ones(Int, v0_block_count), sites, "+", "-")
@show inner(phi, v0)
# The state phi is used as initial state for QKD and DMRG.
# The state v0 (blocked GHZ) is used for the time reversal.

# Verify that v0 is invariant to the operator (I+T)/2
xi_recip_sq = norm(apply(lgtz2_chain_time_rev_mpo(sites, 1), v0))^2	# 1/xi^2
@assert isapprox(xi_recip_sq, 1, atol=1e-5)
@show xi_recip_sq
v0_perp = nothing

# Observable to be projected onto the Krylov basis to check the gauge invariant
obs = [lgtz2_gauge_mpo(sites, 0)]

# Warning, the definitions of H_0 and H_d must match the trotter_lgtz2_chain definitions!
h_0, h_d = lgtz2_chain_mpo(0, 0, sites; j=1), lgtz2_chain_mpo(1, 1, sites; j=0)
time_r_op = lgtz2_chain_time_rev_mpo(sites)
gauge_penalty = gauge_sector>0 ? 1. : 50
additional_ham_terms = [gauge_penalty * lgtz2_gauge_mpo(sites, gauge_sector)]

# Trotter steps implementations.
kry_dt = 0.005
# Note the 1/2 for the case of the time reversal (mystep_tr) as a result of the
# halved time evolution.
mystep_tr(psi::MPS, g::Real) = trotter_lgtz2_chain(g, g, kry_dt / 2, psi, sites; trotter_config...)
mystep_(psi::MPS, g::Real) = trotter_lgtz2_chain(g, g, kry_dt, psi, sites; trotter_config...)

function lgt_check()
    plus = MPS(Float32, sites, "+")
    minus = MPS(Float32, sites, "-")
    @show inner(plus', h_0, plus), inner(plus', h_d, plus)
    @show inner(minus', h_0, minus), inner(minus', h_d, minus)
end

lgt_check()

# DMRG config
sweeps = Sweeps(
  [
    "maxdim" "mindim" "cutoff" "noise"
    50 10 1e-12 1E-7
    100 20 1e-12 1E-8
    200 20 1e-12 1E-10
    400 20 1e-12 0
    800 20 1e-12 1E-11
    800 20 1e-12 0
  ]
)
setmaxdim!(sweeps, 10, 20, 100, 200, 500)
setcutoff!(sweeps, 1e-15)

