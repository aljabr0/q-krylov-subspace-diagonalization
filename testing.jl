# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import LinearAlgebra: eigvals, eigvecs, hermitianpart, Hermitian, I, Diagonal, norm
import Random
include("KTR.jl")
using .KTR
include("KTRMPS.jl")
using .KTRMPS
import ITensorMPS: MPS, MPO, siteinds, inner, random_mps
using Test

rnd_hermitian(n; T=ComplexF64) = hermitianpart(randn(T, (n, n)))

# Build the dense matrix representation of an MPO in the computational
# ("0"/"1") basis, ordered so that site 1 is the most significant bit —
# matching the qubit-index convention used by KTR.pauli_xx/pauli_z.
function mpo_to_matrix(mpo::MPO, sites)
    n = length(sites)
    basis = [MPS(sites, [c == '0' ? "0" : "1" for c in string(i, base=2, pad=n)]) for i=0:(2^n - 1)]
    return [inner(basis[i + 1]', mpo, basis[j + 1]) for i=0:(2^n - 1), j=0:(2^n - 1)]
end

# Same basis convention as mpo_to_matrix, but for an MPS: returns its dense
# state-vector representation.
function mps_to_vector(psi::MPS, sites)
    n = length(sites)
    basis = [MPS(sites, [c == '0' ? "0" : "1" for c in string(i, base=2, pad=n)]) for i=0:(2^n - 1)]
    return [inner(basis[i + 1], psi) for i=0:(2^n - 1)]
end

# Exact exp(-itH) propagator for a dense Hermitian Hamiltonian, via eigendecomposition.
function exact_propagator(H, t)
    vals, vecs = eigvals(H), eigvecs(H)
    return vecs * Diagonal(exp.(-im .* t .* vals)) * vecs'
end

@testset "ising-annni" begin
    for h=range(0, 1, 4), n=[4, 6]
        @test all(+(annni(0., h, n)...) ≈ ising_chain(1, h, n))
    end

    # n=2 is exactly solvable: H = -j X⊗X - h(Z⊗I + I⊗Z) splits into two 2x2
    # blocks with eigenvalues ±j and ±sqrt(j^2 + 4h^2).
    for j=[0.5, 1., 2.], h=[0., 0.3, 1.]
        expected = sort([j, -j, sqrt(j^2 + 4h^2), -sqrt(j^2 + 4h^2)])
        @test eigvals(ising_chain(j, h, 2)) ≈ expected atol=1e-10
    end

    # Global spin flip (X on every site) leaves the XX coupling invariant and
    # flips the sign of the field term, so the spectrum must be even in h.
    for j=[0.5, 1.], h=[0.3, 0.7, 1.5], n=[4, 6]
        @test eigvals(ising_chain(j, h, n)) ≈ eigvals(ising_chain(j, -h, n)) atol=1e-8
    end

    # A staggered Z rotation (Z on alternating sites) flips the sign of every
    # XX bond while leaving the field term invariant, so the spectrum is even in j.
    for j=[0.5, 1.], h=[0.3, 0.7, 1.5], n=[4, 6]
        @test eigvals(ising_chain(j, h, n)) ≈ eigvals(ising_chain(-j, h, n)) atol=1e-8
    end

    # Chains below the minimum size should raise assertion errors.
    @test_throws AssertionError ising_chain(1, 0.5, 1)
    @test_throws AssertionError annni(0., 0.5, 3)
end

@testset "ising_chain_mpo" begin
    # The MPO, expanded in the computational basis, must reproduce the dense
    # reference Hamiltonian from KTR.ising_chain exactly.
    for n=[2, 3, 4], j=[0.5, 1., 2.], h=[0., 0.3, 1.]
        sites = siteinds("S=1/2", n)
        mat = mpo_to_matrix(ising_chain_mpo(j, h, sites), sites)
        @test mat ≈ Matrix(ising_chain(j, h, n)) atol=1e-10
        @test mat ≈ mat' atol=1e-10 # Hermitian
    end

    # n=2 is exactly solvable (see the "ising-annni" testset above).
    for j=[0.5, 1., 2.], h=[0., 0.3, 1.]
        sites = siteinds("S=1/2", 2)
        expected = sort([j, -j, sqrt(j^2 + 4h^2), -sqrt(j^2 + 4h^2)])
        @test eigvals(mpo_to_matrix(ising_chain_mpo(j, h, sites), sites)) ≈ expected atol=1e-10
    end

    # h=0 leaves only the XX coupling (odd under a global spin flip is irrelevant
    # here since it's diagonal in the flip eigenbasis); j=0 leaves only the field.
    let sites = siteinds("S=1/2", 4)
        @test mpo_to_matrix(ising_chain_mpo(1., 0., sites), sites) ≈ Matrix(ising_chain(1., 0., 4)) atol=1e-10
        @test mpo_to_matrix(ising_chain_mpo(0., 1., sites), sites) ≈ Matrix(ising_chain(0., 1., 4)) atol=1e-10
    end

    # Degenerate single-site chain: no XX term is possible, so the MPO reduces
    # to the bare field term -h*Z (unlike KTR.ising_chain, which asserts n >= 2).
    let sites = siteinds("S=1/2", 1)
        mat = mpo_to_matrix(ising_chain_mpo(1., 0.5, sites), sites)
        @test mat ≈ [-0.5 0.; 0. 0.5] atol=1e-10
    end
end

@testset "trotter_ising" begin
    # trotter_ising hardcodes j=1, so its exact reference Hamiltonian is
    # KTR.ising_chain(1, g, n). The Trotterized evolution should converge to the
    # exact exp(-itH) propagator as steps_per_unit_t grows; first order converges
    # more slowly than second order, so it gets a looser tolerance.
    Random.seed!(1)
    for n=[2, 3, 4], g=[0., 0.5, 1.2], t=[0.2, 0.5]
        sites = siteinds("S=1/2", n)
        psi = random_mps(sites; linkdims=2)
        vexact = exact_propagator(ising_chain(1, g, n), t) * mps_to_vector(psi, sites)

        for (order, atol)=[(1, 5e-3), (2, 1e-4)]
            res = trotter_ising(g, t, deepcopy(psi), sites; steps_per_unit_t=300, order=order)
            @test mps_to_vector(res, sites) ≈ vexact atol=atol
        end
    end

    # At a fixed (coarse) step size, second-order (Strang) splitting should be
    # markedly more accurate than first-order splitting.
    let n=4, g=0.8, t=0.5, spu=30
        sites = siteinds("S=1/2", n)
        psi = random_mps(sites; linkdims=2)
        vexact = exact_propagator(ising_chain(1, g, n), t) * mps_to_vector(psi, sites)

        err(order) = norm(mps_to_vector(trotter_ising(g, t, deepcopy(psi), sites; steps_per_unit_t=spu, order=order), sites) - vexact)
        @test err(2) < err(1)
    end

    # t=0 is a no-op evolution (regression test: trotter_factory used to crash for
    # order=2 at t=0, since trotter_step returns steps=0 there and the general
    # branch computed repeat(gates, outer=steps-1) with a negative outer count).
    for n=[2, 3], g=[0., 0.7], order=(1, 2)
        sites = siteinds("S=1/2", n)
        psi = random_mps(sites; linkdims=2)
        res = trotter_ising(g, 0.0, deepcopy(psi), sites; order=order)
        @test inner(res, psi) ≈ 1 atol=1e-10
    end

    # The evolution is unitary, so (well within the default truncation cutoff at
    # these small system sizes) the resulting MPS should remain normalised.
    for n=[2, 3, 4], g=[0., 0.7], t=[0.1, 1.0], order=(1, 2)
        sites = siteinds("S=1/2", n)
        psi = random_mps(sites; linkdims=2)
        res = trotter_ising(g, t, psi, sites; order=order)
        @test inner(res, res) ≈ 1 atol=1e-8
    end

    # H = -∑ᵢXᵢXᵢ₊₁ - g∑ᵢZᵢ (j=1, matching trotter_ising) commutes with its own
    # exponential, so the energy expectation value is conserved under evolution.
    for n=[3, 4], g=[0.3, 0.9]
        sites = siteinds("S=1/2", n)
        h_mpo = ising_chain_mpo(1., g, sites)
        psi = random_mps(sites; linkdims=2)
        e0 = real(inner(psi', h_mpo, psi))
        res = trotter_ising(g, 0.6, psi, sites; steps_per_unit_t=200)
        e1 = real(inner(res', h_mpo, res))
        @test e1 ≈ e0 atol=1e-3
    end
end

@testset "mps_ghz_blocks" begin
    # For every choice of (c, n, signs) the resulting state should be normalised and
    # supported exactly on the 2^c computational basis states obtained by setting each
    # of the c blocks entirely to state0 ("0") or entirely to state1 ("1"), each with
    # amplitude ±1/sqrt(2^c) according to the product of the corresponding signs.
    for c=1:3, blocks_per=[1, 2]
        n = c * blocks_per
        sites = siteinds("S=1/2", n)
        for signs_bits=0:(2^c - 1)
            signs = [isodd(signs_bits >> (i - 1)) ? -1 : 1 for i=1:c]
            psi = mps_ghz_blocks(signs, sites)

            @test inner(psi, psi) ≈ 1 atol=1e-6

            for i=0:(2^c - 1)
                pattern = digits(i, base=2, pad=c)
                bsign = prod(signs .^ pattern)
                basis_str = join(string(b == 0 ? "0" : "1")^blocks_per for b in pattern)
                basis_state = MPS(sites, split(basis_str, ""))
                @test inner(basis_state, psi) ≈ bsign / sqrt(2^c) atol=1e-6
            end

            # A basis state that mixes state0/state1 within a single block (only
            # possible when blocks have more than one site) has zero overlap.
            if blocks_per > 1
                mixed_str = "0" * "1"^(n - 1)
                mixed_state = MPS(sites, split(mixed_str, ""))
                @test inner(mixed_state, psi) ≈ 0 atol=1e-6
            end
        end
    end

    # Custom single-character state labels (repeated to fill a block) should also
    # produce a normalised state.
    sites = siteinds("S=1/2", 4)
    psi = mps_ghz_blocks([1, -1], sites, "+", "-")
    @test inner(psi, psi) ≈ 1 atol=1e-6

    # Invalid inputs should raise assertion errors.
    sites = siteinds("S=1/2", 4)
    @test_throws AssertionError mps_ghz_blocks([2], sites)       # sign not ±1
    @test_throws AssertionError mps_ghz_blocks([1, 1, 1], sites) # n % c != 0
    @test_throws AssertionError mps_ghz_blocks(ones(Int, 9), siteinds("S=1/2", 9)) # c > 8
end
