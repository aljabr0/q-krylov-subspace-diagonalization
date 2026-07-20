# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

module KTRMPS
import ITensorMPS: MPS, MPO, apply, siteinds, OpSum, inner, op
import LinearAlgebra: Hermitian, normalize!

include("KTR.jl")
import .KTR: triu2h, symmtoe

export ising_chain_mpo, overlap_mps, overlap_mps_toe
export overlap_time_r, overlaps_time_r_with_sign, overlaps_time_r_with_sign_no_gram, time_r_get_sign
export krylov_seq
export trotter_ising
export lgtz2_chain_mpo, trotter_lgtz2_chain
export ising_chain_time_rev_mpo, lgtz2_chain_time_rev_mpo, lgtz2_gauge_mpo
export trotter_annni, annni_mpo
export mps_ghz_blocks

CUTOFF_DEFAULT = 1e-15

"""
    ising_chain_mpo(j, h, sites)

Construct the MPO for the transverse-field Ising chain Hamiltonian
    H = -j ∑ᵢ XᵢXᵢ₊₁ - h ∑ᵢ Zᵢ.

# Arguments
- `j`: nearest-neighbour XX coupling strength.
- `h`: transverse field strength.
- `sites`: ITensor site indices defining the chain.
"""
function ising_chain_mpo(j::Real, h::Real, sites)
    opsum = OpSum()
    for i=eachindex(sites)[begin:(end - 1)]
        opsum += -j,"X",i,"X",i+1
        opsum += -h,"Z",i
    end
    opsum += -h,"Z",size(sites, 1)
    return MPO(opsum, sites)
end

"""
    ising_chain_time_rev_mpo(sites, proj_s=0)

Construct the time-reversal operator for the transverse-field Ising chain as an MPO.
The operator is a product of alternating Y and X Pauli matrices.

# Arguments
- `sites`: ITensor site indices (must have even length).
- `proj_s`: if `0` (default), return the full time-reversal MPO; if `±1`, return the
  orthogonal projector onto the eigenspace with eigenvalue `proj_s`.
"""
function ising_chain_time_rev_mpo(sites, proj_s=0)
    @assert proj_s in (0, 1, -1)
    pattern = repeat(["Y", "X"], trunc(Int, size(sites, 1) / 2))
    if proj_s == 0
	return MPO(ComplexF64, sites, pattern)
    end
    # With proj_j either 1 or -1 return a projector related to the corresponding
    # eigenspace.
    return (proj_s * MPO(ComplexF64, sites, pattern) + MPO(ComplexF64, sites, "I"))/2
end

"""
    count_lgtz2_zzz_blocks(n)

Return the number of ZZZ interaction blocks in a Z₂ lattice gauge theory chain of
`n` sites. If `n` is even it is reduced to `n-1` first to align with the
odd-site lattice structure.
"""
function count_lgtz2_zzz_blocks(n)
    n = n % 2 == 0 ? n-1 : n
    return Int((n-1) / 2)
end

"""
    lgtz2_chain_mpo(m, g, sites; j=1)

Construct the MPO for the (1+1)D Z₂ lattice gauge theory Hamiltonian
    H = -j ∑ ZᵢZᵢ₊₁Zᵢ₊₂ - g ∑ Xₗᵢₙₖ - m ∑ Xₘₐₜₜₑᵣ.

The chain alternates between link and matter sites; ZZZ plaquettes span every
link–matter–link triplet.

# Arguments
- `m`: mass parameter (coupling on matter X terms).
- `g`: gauge coupling (coupling on link X terms).
- `sites`: ITensor site indices (odd-length chains expected).
- `j`: ZZZ coupling strength (default `1`).
"""
function lgtz2_chain_mpo(m::Real, g::Real, sites; j=oneunit(m))
    opsum = OpSum()
    last_zzz_base_i = -1
    for i=eachindex(sites)[begin:2:(end - 2)]
        opsum += -j,"Z",i,"Z",i+1,"Z",i+2
        opsum += -g,"X",i	# Start with link
        opsum += -m,"X",i+1
	last_zzz_base_i = i
    end
    @assert last_zzz_base_i >= 1
    @assert last_zzz_base_i % 2 == 1
    opsum += -g,"X",last_zzz_base_i+2	# Last link
    @assert last_zzz_base_i+2 == (count_lgtz2_zzz_blocks(size(sites, 1))) * 2 + 1

    return MPO(opsum, sites)
end

"""
    lgtz2_gauge_mpo(sites, sign=0)

Construct the averaged gauge-operator MPO for the Z₂ lattice gauge theory chain.
The gauge operator is the normalised sum of all XXX stabilisers on internal links.

# Arguments
- `sites`: ITensor site indices.
- `sign`: if `0` (default), return the bare gauge MPO; if `±1`, return a PSD projector
  that penalises the sector with eigenvalue `-sign`.
"""
function lgtz2_gauge_mpo(sites, sign=0)
    # Averaged sum of the gauge operators
    @assert sign in (0, 1, -1)
    opsum = OpSum()
    c = 0
    f = 1 / (count_lgtz2_zzz_blocks(size(sites, 1)) - 1)
    for i=eachindex(sites)[2:2:(end - 3)]
        opsum += f,"X",i,"X",i+1,"X",i+2
	c += 1
    end

    @assert count_lgtz2_zzz_blocks(size(sites, 1)) - 1 == c
    sign == 0 && return MPO(opsum, sites)
    # Return PSD panalizing the sector corresponding to the eigenvalue -sign
    return -sign/2 * MPO(opsum, sites) + MPO(ComplexF64, sites, "I") / 2
end

"""
    lgtz2_chain_time_rev_mpo(sites, proj_s=0)

Construct the time-reversal operator for the Z₂ lattice gauge theory chain as an MPO
(tensor product of Y on every site).

# Arguments
- `sites`: ITensor site indices.
- `proj_s`: if `0` (default), return the full time-reversal MPO; if `±1`, return the
  orthogonal projector onto the eigenspace with eigenvalue `proj_s`.
"""
function lgtz2_chain_time_rev_mpo(sites, proj_s=0)
    @assert proj_s in (0, 1, -1)
    if proj_s == 0
	return MPO(ComplexF64, sites, "Y")
    end
    return (proj_s * MPO(ComplexF64, sites, "Y") + MPO(ComplexF64, sites, "I"))/2
end

# Given the total time and the steps per unit of time, produce the number of required steps
# and the time per step.
"""
    trotter_step(total_t, steps_per_unit_t)

Compute the number of Trotter steps and the time increment per step.

Returns `(0, zero(total_t))` when `total_t ≈ 0`. Otherwise rounds
`|total_t| * steps_per_unit_t` to the nearest integer (minimum 1) and sets
`tau = total_t / steps`, preserving the sign of `total_t`.

# Arguments
- `total_t`: total evolution time (may be negative for backward evolution).
- `steps_per_unit_t`: desired number of Trotter steps per unit of time.

# Returns
`(steps::Int, tau)` — number of steps and time per step.
"""
function trotter_step(total_t::Real, steps_per_unit_t::Int)
    if isapprox(total_t, 0)
        return 0, zero(total_t)
    end
    steps = round(Int, abs(total_t) * steps_per_unit_t)
    steps = max(steps, 1)
    tau = total_t / steps
    return steps, tau
end

"""
    trotter_factory(hops_a, hops_b, t=1.0; order=2, steps_per_unit_t=10)

Build the ordered sequence of local gates for Suzuki–Trotter time evolution of
`exp(-it(A+B))`, where A and B are each split into commuting sub-terms.

# Arguments
- `hops_a`, `hops_b`: vectors of local Hermitian operators representing the two
  parts of the Hamiltonian split (each term is exponentiated independently).
- `t`: total evolution time.
- `order`: Trotter order — `1` (first-order) or `2` (second-order Strang splitting,
  default).
- `steps_per_unit_t`: number of Trotter steps per unit time.

# Returns
A flat vector of gate matrices ready to be passed to `ITensorMPS.apply`.
"""
function trotter_factory(hops_a, hops_b, t::Real=1.0; order::Int=2, steps_per_unit_t=10)
    @assert order in (1, 2)
    steps, tau = trotter_step(t, steps_per_unit_t)
    gates_b = [exp(-im * tau * hop) for hop=hops_b]
    steps == 0 && return empty(gates_b)
    gates = nothing
    if order == 2
	# Cases for steps: 1, 2
	gates_ah = [exp(-im * tau/2 * hop) for hop=hops_a]
	steps == 1 && return [gates_ah; gates_b; gates_ah]
	gates_a = [exp(-im * tau * hop) for hop=hops_a]
	steps == 2 && return [gates_ah; gates_b; gates_a; gates_b; gates_ah]

	# Case steps >= 3
	gates = [gates_b; gates_a]
	gates = repeat(gates, outer=steps-1)
	return [gates_ah; gates; gates_b; gates_ah]
    end
    gates_a = [exp(-im * tau * hop) for hop=hops_a]
    gates = [gates_a; gates_b]
    return repeat(gates, outer=steps)
end

"""
    trotter_ising(g, t, psi, sites; cutoff=CUTOFF_DEFAULT, steps_per_unit_t=10, max_bond_dim=100, order=2)

Apply Suzuki–Trotter time evolution `exp(-itH)` to the MPS `psi` under the
transverse-field Ising Hamiltonian H = -∑ XᵢXᵢ₊₁ - g ∑ Zᵢ.

# Arguments
- `g`: transverse field strength.
- `t`: total evolution time.
- `psi`: initial MPS state.
- `sites`: ITensor site indices.
- `cutoff`: singular-value cutoff for MPS compression.
- `steps_per_unit_t`: number of Trotter steps per unit time.
- `max_bond_dim`: maximum MPS bond dimension after each gate application.
- `order`: Trotter order (`1` or `2`).

# Returns
The time-evolved MPS.
"""
function trotter_ising(g::Real, t::Real, psi::MPS, sites;cutoff=CUTOFF_DEFAULT, steps_per_unit_t=10, max_bond_dim=100, order=2)
    s, n = sites, size(sites, 1)
    hops_a = [-1 * op("X", s[i]) * op("X", s[i + 1]) for i=1:(n - 1)]
    hops_b = [-g * op("Z", s[i]) for i=1:n]
    gates = trotter_factory(hops_a, hops_b, t; steps_per_unit_t=steps_per_unit_t, order=order)
    return apply(gates, psi; cutoff=cutoff, maxdim=max_bond_dim)
end

"""
    trotter_lgtz2_chain(m, g, t, psi, sites; cutoff=CUTOFF_DEFAULT, steps_per_unit_t=10, max_bond_dim=100, order=2, j=1)

Apply Suzuki–Trotter time evolution to the MPS `psi` under the Z₂ lattice gauge
theory Hamiltonian (ZZZ plaquette + X matter/link terms).

# Arguments
- `m`: mass coupling (matter X terms).
- `g`: gauge coupling (link X terms).
- `t`: total evolution time.
- `psi`: initial MPS state.
- `sites`: ITensor site indices.
- `cutoff`: singular-value cutoff for MPS compression.
- `steps_per_unit_t`: number of Trotter steps per unit time.
- `max_bond_dim`: maximum MPS bond dimension.
- `order`: Trotter order (`1` or `2`).
- `j`: ZZZ coupling strength (default `1`).

# Returns
The time-evolved MPS.
"""
function trotter_lgtz2_chain(m::Real, g::Real, t::Real, psi::MPS, sites;cutoff=CUTOFF_DEFAULT, steps_per_unit_t=10, max_bond_dim=100, order=2, j=1)
    s, n = sites, size(sites, 1)
    @assert n > 1
    hops_a = [-j * op("Z", s[i]) * op("Z", s[i + 1]) * op("Z", s[i + 2]) for i=1:(n - 2)]

    hops_b = nothing
    let c
	c = count_lgtz2_zzz_blocks(size(sites, 1)) * 2 + 1
	hops_b = [-(i % 2 == 0 ? m : g) * op("X", s[i]) for i=1:c]
    end

    gates = trotter_factory(hops_a, hops_b, t; steps_per_unit_t=steps_per_unit_t, order=order)
    return apply(gates, psi; cutoff=cutoff, maxdim=max_bond_dim)
end

"""
    annni_mpo(kappa, h, sites; j=1)

Construct the Hamiltonian MPOs for the Axial Next-Nearest-Neighbour Ising (ANNNI)
model
    H = -J₁ ∑ XᵢXᵢ₊₁ + J₂ ∑ XᵢXᵢ₊₂ - B ∑ Zᵢ

where J₂ = κ·J₁ and B = h·J₁. The Hamiltonian is returned as three separate MPOs
so that each driving direction can be scaled independently.

# Arguments
- `kappa`: frustration parameter κ = J₂/J₁.
- `h`: dimensionless transverse field B/J₁.
- `sites`: ITensor site indices (must satisfy `length ≥ 4` and `length % 2 == 0`).
- `j`: overall coupling scale J₁ (default `1`).

# Returns
`(h_nn, h_nnn, h_z)` — MPOs for the nearest-neighbour XX term, the
next-nearest-neighbour XX term, and the transverse-field Z term, respectively.
"""
function annni_mpo(kappa::Real, h::Real, sites; j=1)
    """
    Construct the Hamitonian (on the given sites) for the ANNNI model defined by the parameters
    kappa and h (external field). Returns 3 MPOs corresponding to the base,
    and the two driving terms for kappa and h, respectively.
    """
    # k=-J_2/J_1, h=B/J_1
    # H = -J_1 H_nn -J_2 H_nnn -B H_z
    @assert length(sites) >= 4 && length(sites) % 2 == 0 && ndims(sites) == 1

    m_j2 = j * kappa # minus j2
    opsum = OpSum()
    for i=eachindex(sites)[begin:(end - 2)]
        opsum += m_j2,"X",i,"X",i+2
    end
    h_nnn = MPO(opsum, sites)

    # Note that ining_chain_mpo multiplies the params by -1
    h_z, h_nn = ising_chain_mpo(0, h * j, sites), ising_chain_mpo(j, 0, sites)
    return h_nn, h_nnn, h_z	# h_0, h_d1, h_d2
end

"""
    trotter_annni(kappa, h, t, psi, sites; cutoff=CUTOFF_DEFAULT, steps_per_unit_t=10, max_bond_dim=100, order=2, j=1)

Apply Suzuki–Trotter time evolution to the MPS `psi` under the ANNNI Hamiltonian
(nearest-neighbour XX, next-nearest-neighbour XX, and transverse-field Z terms).

# Arguments
- `kappa`: frustration parameter κ = J₂/J₁.
- `h`: dimensionless transverse field B/J₁.
- `t`: total evolution time.
- `psi`: initial MPS state.
- `sites`: ITensor site indices.
- `cutoff`: singular-value cutoff for MPS compression.
- `steps_per_unit_t`: number of Trotter steps per unit time.
- `max_bond_dim`: maximum MPS bond dimension.
- `order`: Trotter order (`1` or `2`).
- `j`: overall coupling scale J₁ (default `1`).

# Returns
The time-evolved MPS.
"""
function trotter_annni(kappa::Real, h::Real, t::Real, psi::MPS, sites;
			cutoff=CUTOFF_DEFAULT, steps_per_unit_t=10, max_bond_dim=100, order=2,
			j=1)
    s, n = sites, size(sites, 1)

    hops_a1 = [-j * op("X", s[i]) * op("X", s[i + 1]) for i=1:(n - 1)]
    hops_a2 = [(j * kappa) * op("X", s[i]) * op("X", s[i + 2]) for i=1:(n - 2)]
    hops_b = [-(h * j) * op("Z", s[i]) for i=1:n]

    gates = trotter_factory([hops_a1; hops_a2], hops_b, t; steps_per_unit_t=steps_per_unit_t, order=order)
    return apply(gates, psi; cutoff=cutoff, maxdim=max_bond_dim)
end

"""
    krylov_seq(step_f, psi_s, n, args...)
    krylov_seq(step_f, psi, n, args...)

Build a block Krylov sequence by repeatedly applying `step_f` to the current block.

Starting from the initial block `psi_s` of `k` states, the function appends
`step_f(v, args...)` for each vector `v` in the previous block until the sequence
spans `n` complete blocks (total length `n * k`).

The single-MPS variant wraps `psi` in a length-1 vector and delegates to the
block form.

# Arguments
- `step_f`: function mapping an MPS to the next Krylov vector.
- `psi_s`: vector of starting MPS states (block size `k = length(psi_s)`).
- `psi`: single starting MPS (shorthand for block size 1).
- `n`: total number of blocks to generate.
- `args...`: additional arguments forwarded verbatim to `step_f`.

# Returns
A `Vector` of MPS of length `n * k`.
"""
function krylov_seq(step_f::Function, psi_s::AbstractVector, n::Int, args...)
    start_c = size(psi_s, 1)   # Krylov block size
    v = copy(psi_s)
    for j=2:n
        b = size(v, 1) - start_c
        for k=1:start_c
            push!(v, step_f(v[b + k], args...))
        end
    end
    return v
end
krylov_seq(step_f::Function, psi::MPS, n::Int, args...) = krylov_seq(step_f, [psi], n, args...)

# TODO use zero() with type inherited from MPS
"""
    overlap_mps(psi_s)

Compute the Hermitian overlap (Gram) matrix S with Sᵢⱼ = ⟨ψᵢ|ψⱼ⟩ for all MPS in
`psi_s`. Only the upper-triangular part is computed; the result is symmetrised via
`triu2h`.
"""
overlap_mps(psi_s::Vector{MPS}) = triu2h(complex.([i<=j ? inner(psi_s[i], psi_s[j]) : 0. for i=axes(psi_s, 1), j=axes(psi_s, 1)]))

"""
    overlap_mps_toe(psi_s)

Compute the symmetric Toeplitz overlap matrix built from the first row
`[⟨ψ₁|ψⱼ⟩ for j = 1, …, n]`.
"""
overlap_mps_toe(psi_s::Vector{MPS}) = symmtoe(complex.([inner(psi_s[1], psi_s[j]) for j=axes(psi_s, 1)]))	# TODO Not compatible with multi-start?

"""
    overlap_mps_toe(psi_s, op)

Compute the symmetric Toeplitz overlap matrix built from the operator-weighted first
row `[⟨ψ₁|O|ψⱼ⟩ for j = 1, …, n]` using the MPO `op`.
"""
overlap_mps_toe(psi_s::Vector{MPS}, op::MPO) = symmtoe(complex.([inner(psi_s[1]', op, psi_s[j]) for j=axes(psi_s, 1)]))	# TODO Not compatible with multi-start?

"""
    overlap_mps(psi_s, op)

Compute the Hermitian operator overlap matrix H with Hᵢⱼ = ⟨ψᵢ|O|ψⱼ⟩ for the MPO
`op`. Only the upper-triangular part is computed; the result is symmetrised via
`triu2h`.
"""
overlap_mps(psi_s::Vector{MPS}, op::MPO) = triu2h(complex.([i<=j ? inner(psi_s[i]', op, psi_s[j]) : 0. for i=axes(psi_s, 1), j=axes(psi_s, 1)]))

# Time reversal overlaps
# TODO Generalize the function below, rename time_r_op -> op
"""
    overlap_time_r(psi_s, time_r_op)

Compute the diagonal time-reversal overlaps `[⟨ψᵢ|T|ψᵢ⟩ for i = 1, …, n]` and
embed them in a symmetric Toeplitz structure via `symmtoe`.
"""
overlap_time_r(psi_s::Vector{MPS}, time_r_op::MPO) = symmtoe([inner(psi_s[i]', time_r_op, psi_s[i]) for i=axes(psi_s, 1)])

"""
    time_r_get_sign(time_r_op, psi_s, sign_c=0)

Determine the time-reversal eigenvalue sign of the first Krylov vector `psi_s[1]`.

If `sign_c == 0` (default), the sign is computed as `sign(⟨ψ₁|T|ψ₁⟩)` and
asserted to be ≈ ±1. A non-zero `sign_c` is returned directly after asserting it
is ±1.

# Arguments
- `time_r_op`: MPO representing the time-reversal operator T.
- `psi_s`: vector of Krylov MPS states (first element is taken as ψ₁).
- `sign_c`: pre-computed sign (`0` triggers auto-detection).

# Returns
`sign_c ∈ {+1, -1}`.
"""
function time_r_get_sign(time_r_op::MPO, psi_s::Vector{MPS}, sign_c=0)
    if sign_c == 0
	# Assume first Krylov vector is v0.
	v0 = psi_s[1]
	sign_c = real(inner(v0', time_r_op, v0))
	@assert isapprox(abs(sign_c), 1.; atol=10e-6)
	sign_c = sign(sign_c)
    end
    @assert sign_c in (1, -1)
    return sign_c
end

"""
    overlaps_time_r_with_sign(hs, time_r_op, psi_s, sign_c=0)

Compute the time-reversal Krylov overlap matrices A and B for the generalised
eigenvalue problem with time-reversal symmetry.

The Hamiltonian is passed as a vector of MPOs `hs` (as in multi-term DMRG) to avoid
forming the full MPO sum. The sign of the initial vector is detected via
`time_r_get_sign` unless `sign_c` is provided.

# Arguments
- `hs`: vector of Hamiltonian MPOs (their sum defines H).
- `time_r_op`: MPO representing the time-reversal operator T.
- `psi_s`: vector of Krylov MPS states.
- `sign_c`: pre-computed time-reversal sign (`0` triggers auto-detection).

# Returns
`(a, b)` — Toeplitz overlap matrices where
`a = -sign_c · ∑ₕ ⟨ψᵢ|T·h|ψᵢ⟩` and `b = sign_c · ⟨ψᵢ|T|ψᵢ⟩`.
"""
function overlaps_time_r_with_sign(hs::Vector{MPO}, time_r_op::MPO, psi_s::Vector{MPS}, sign_c=0)
    # The Hamiltonian is passed as list of MPOs (like DMRG in iTensor) so one can avoid summing up MPOs.
    sign_c = time_r_get_sign(time_r_op, psi_s, sign_c)
    @assert sign_c in (1, -1)
    # TODO use inner(B::MPO, y::MPS, A::MPO, x::MPS) instead of apply(...)
    a = sum(-sign_c * overlap_time_r(psi_s, apply(h, time_r_op; alg="naive", truncate=false)) for h in hs)
    b = sign_c * overlap_time_r(psi_s, time_r_op)
    return a, b
end

"""
    overlaps_time_r_with_sign_no_gram(hs, time_r_op, psi_s, sign_c=0)

Like `overlaps_time_r_with_sign` but returns only the Hamiltonian overlap matrix A,
omitting the Gram matrix B.

# Arguments
- `hs`: vector of Hamiltonian MPOs.
- `time_r_op`: MPO representing the time-reversal operator T.
- `psi_s`: vector of Krylov MPS states.
- `sign_c`: pre-computed time-reversal sign (`0` triggers auto-detection).

# Returns
`a` — Toeplitz overlap matrix `a = -sign_c · ∑ₕ ⟨ψᵢ|T·h|ψᵢ⟩`.
"""
function overlaps_time_r_with_sign_no_gram(hs::Vector{MPO}, time_r_op::MPO, psi_s::Vector{MPS}, sign_c=0)
    # The Hamiltonian is passed as list of MPOs (like DMRG in iTensor) so one can avoid summing up MPOs.
    sign_c = time_r_get_sign(time_r_op, psi_s, sign_c)
    @assert sign_c in (1, -1)
    # TODO use inner(B::MPO, y::MPS, A::MPO, x::MPS) instead of apply(...)
    return sum(-sign_c * overlap_time_r(psi_s, apply(h, time_r_op; alg="naive", truncate=false)) for h in hs)
end

function _repeat_pattern(pattern::String, req_len::Int)
    length(pattern) == req_len && return pattern
    pattern = repeat(pattern, div(req_len, length(pattern)))
    @assert length(pattern) == req_len
    return pattern
end

"""
    mps_ghz_blocks(signs, sites, state0="0", state1="1")

Construct a normalised GHZ-like MPS that is an equal superposition over all `2^c`
computational basis states formed by concatenating `c` blocks, each block being
either `state0` or `state1`, weighted by the product of `signs`.

Formally, with `c = length(signs)` and block size `bsz = n / c`:
    |ψ⟩ ∝ ∑_{b ∈ {0,1}^c} (∏ᵢ signsᵢ^{bᵢ}) |block(b₁) ⊗ ⋯ ⊗ block(bₙ)⟩.

# Arguments
- `signs`: length-`c` vector of ±1 values, one per block (`c ≤ 8`).
- `sites`: ITensor site indices (length must be divisible by `c`).
- `state0`: single-site state label for the "0" block; repeated to fill a block.
- `state1`: single-site state label for the "1" block; repeated to fill a block.

# Returns
A normalised `MPS` (Float32 entries).
"""
function mps_ghz_blocks(signs::Vector{Int}, sites, state0="0", state1="1")
    # signs: signs for the excited state "1"
    c, n = size(signs, 1), size(sites, 1)
    @assert all(abs.(signs) .== 1)
    @assert c <= 8 # This code is not efficient for large compositions
    @assert n % c == 0
    bsz = div(n, c)
    ret = MPS[]

    state0 = _repeat_pattern(state0, bsz)
    state1 = _repeat_pattern(state1, bsz)
    states = [state0, state1]

    for i=0:(2^c-1)
	pattern = digits(i, base=2, pad=c)
	bsign = prod(signs .^ pattern)
	pattern = string(states[pattern .+ 1]...)
	pattern = split(pattern, "")
	push!(ret, bsign * MPS(Float32, sites, pattern))
    end
    ret = reduce(+, ret)
    normalize!(ret)
    return ret
end

# End of module
end;
