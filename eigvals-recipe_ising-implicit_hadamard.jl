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

n = 64
krylov_size = 128
trotter_config = (cutoff=1e-15, steps_per_unit_t=500, max_bond_dim=500, order=2)
gammas = range(0.01, .1, 32) # Sample of parameters

sites = siteinds("S=1/2", n)

# The state phi is used as initial state for QKD and DMRG.
# The states v0 and v0_perp are used for the time reversal.
phi = MPS(Float32, sites, "+")
normalize!(phi)

v0 = apply(ising_chain_time_rev_mpo(sites, 1), phi)
xi_recip_sq = norm(v0)^2
@show xi_recip_sq
v0_perp = apply(ising_chain_time_rev_mpo(sites, -1), phi)
@show norm(v0), norm(v0_perp)
normalize!(v0)	# TODO Veify norm first!
normalize!(v0_perp)	# TODO Veify norm first!

# Verify v0 and v0_perp are stabilized by T
@assert isapprox(inner(v0', ising_chain_time_rev_mpo(sites, 0), v0), 1., atol=1e-5)
@assert isapprox(inner(v0_perp', ising_chain_time_rev_mpo(sites, 0), v0_perp), -1., atol=1e-5)

model_name = "Ising-implicit_hadamard"

# Warning, the definitions of H_0 and H_d must match the trotter_ising definitions!
h_0, h_d = ising_chain_mpo(1, 0, sites), ising_chain_mpo(0, 1, sites)
time_r_op = ising_chain_time_rev_mpo(sites)

# Trotter steps implementations.
kry_dt = 0.01
# Note the 1/2 for the case of the time reversal (mystep_tr) as a result of the
# halved time evolution.
mystep_tr(psi::MPS, g::Real) = trotter_ising(g, kry_dt / 2, psi, sites; trotter_config...)
mystep_(psi::MPS, g::Real) = trotter_ising(g, kry_dt, psi, sites; trotter_config...)

