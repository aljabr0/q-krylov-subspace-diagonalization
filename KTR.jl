# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

module KTR

import LinearAlgebra: I, Hermitian, dot, eigen, eigvecs, diagm, diag
import LinearAlgebra: norm, triu
import Infinities: RealInfinity

export op_u2, pauli_xx, pauli_z, ising_chain, annni
export eval_ham, Ham
export krylov_seq, krylov_matrix
export triu2h, symmtoe, proj_psd, spectral_thr

ScalarOrVec = Union{RealInfinity, Number, Array{<:Number, 1}}

"""
    spectral_thr(mat::Hermitian, thr::Real) -> (eigenvalues, eigenvectors)

Return the eigenvalues and corresponding eigenvectors of `mat` whose eigenvalues
exceed `thr`.
"""
function spectral_thr(mat::Hermitian, thr::Real)
    l, u = eigen(mat)
    u = u[:, l .> thr]
    l = l[l .> thr]
    return l, u
end

"""
    triu2h(m::AbstractMatrix) -> Hermitian

Symmetrize the upper-triangular part of `m` into a Hermitian matrix.
The diagonal is taken as-is; the lower triangle is set to the conjugate of the
upper triangle.
"""
triu2h(m::AbstractMatrix)::Hermitian = Hermitian(triu(m, 1) + triu(m, 1)' + diagm(diag(m)))

"""
    symmtoe(v::AbstractVector) -> Hermitian

Construct a Hermitian Toeplitz matrix from first row `v`.
The imaginary part of `v[1]` is discarded to enforce a real diagonal.
"""
function symmtoe(v::AbstractVector)
    n = size(v, 1)
    pairs_u = [Pair(j - 1, fill(v[j], n - j + 1)) for j=1:n]
    pairs_l = [Pair(1 - j, fill(conj(v[j]), n - j + 1)) for j=2:n]
    return Hermitian(diagm([pairs_u; pairs_l]...))
end

"""
    proj_psd(h::Hermitian, spectrum_shift=0) -> Hermitian

Project `h` onto the set of positive semi-definite matrices by clamping all
negative eigenvalues to zero. An optional `spectrum_shift` is then added to
every eigenvalue.
"""
function proj_psd(h::Hermitian, spectrum_shift=0)::Hermitian
    l, u = eigen(h)
    l = max.(l, zero(first(l))) .+ spectrum_shift
    return Hermitian(u * diagm(l) * u')
end

"""
    pauli_xx(i::Int, j::Int, n::Int) -> Hermitian

Construct the Pauli XX operator acting on qubits `i` and `j` (0-indexed) within
an `n`-qubit system.
"""
function pauli_xx(i::Int, j::Int, n::Int)
    mat_x = Matrix([0. 1.;1. 0.])
    @assert i != j
    i, j = sort([i, j])
    @assert i >= 0 && j < n
    m1 = Matrix(1.0I, 2^i, 2^i)
    m2 = Matrix(1.0I, 2^(j - i - 1), 2^(j - i - 1))
    m3 = Matrix(1.0I, 2^(n - j - 1), 2^(n - j - 1))
    return Hermitian(reduce(kron, [m1, mat_x, m2, mat_x, m3]))
end

"""
    op_u2(i::Int, u::AbstractMatrix, n::Int) -> Hermitian

Construct the matrix corresponding to the local 2×2 operator `u` applied to
qubit `i` (0-indexed) within an `n`-qubit system.
"""
function op_u2(i::Int, u::AbstractMatrix, n::Int)
    @assert i >= 0 && i < n
    @assert size(u) == (2, 2)
    m1 = Matrix(I, 2^i, 2^i)
    m2 = Matrix(I, 2^(n - i - 1), 2^(n - i - 1))
    m1, u, m2 = promote(m1, u, m2)
    return Hermitian(reduce(kron, [m1, u, m2]))
end

"""
    pauli_z(i, n) -> Hermitian

Construct the Pauli Z operator acting on qubit `i` (0-indexed) within an
`n`-qubit system.
"""
pauli_z(i, n) = op_u2(i, [1 0;0 -1], n)

"""
    Ham

Hamiltonian with a base term `h_0` and a driver term `h_d`, parametrised by
`gamma`, so that `H = h_0 + gamma * h_d`.

`gamma` can be a scalar, `RealInfinity`, or a vector of coefficients (in which
case `h_d` should be an iterable of matching Hermitian terms).
"""
struct Ham
    gamma::ScalarOrVec
    h_0::Hermitian
    h_d
end

"""
    eval_ham(gamma, h_0, h_d) -> Hermitian
    eval_ham(ham::Ham)        -> Hermitian

Evaluate the Hamiltonian `H = h_0 + gamma * h_d`.

- Scalar `gamma`: returns `h_0 + gamma * h_d`.
- `RealInfinity` `gamma`: returns `sign(gamma) * h_d` (only the driver term survives).
- Vector `gamma`: returns `h_0 + Σᵢ gamma[i] * h_d[i]`.
"""
eval_ham(gamma::Number, h_0::Hermitian, h_d::Hermitian) = Hermitian(h_0 .+ (gamma .* h_d))
eval_ham(gamma::RealInfinity, h_0::Hermitian, h_d::Hermitian) = Hermitian(sign(gamma) * h_d)
#eval_ham(gamma::Array{<:Number, 1}, h_0::Hermitian, h_d) = h_0 .+ sum([c_i .* h_d_i for (k, h)=zip(gamma, h_d)])
eval_ham(ham::Ham) = eval_ham(ham.gamma, ham.h_0, ham.h_d)

"""
    ising_chain(j, h, n::Int) -> Hermitian

Construct the Ising chain Hamiltonian `H = -j ΣXX - h ΣZ` on `n` qubits.

`j` is the nearest-neighbor coupling strength and `h` is the transverse field.
"""
function ising_chain(j, h, n::Int)
    @assert n >= 2
    m1 = sum(ntuple(k -> pauli_xx(k - 1, k, n), n - 1))
    m2 = sum(ntuple(k -> pauli_z(k - 1, n), n))
    #if boundary == BoundaryPeriodic
    #    m1 += pauli_xx(0, n - 1, n)
    #end
    return Hermitian((-j) .* m1 + (-h) .* m2)
end

"""
    annni(kappa, h, n::Int; j=1) -> (H_nn, H_nnn, H_z)

Construct the Axial Next-Nearest-Neighbor Ising (ANNNI) model Hamiltonian terms
on `n` qubits (requires `n ≥ 4`).

Returns three Hermitian matrices:
- `H_nn`:  nearest-neighbor XX coupling, scaled by `-j`
- `H_nnn`: next-nearest-neighbor XX coupling, scaled by `j * kappa` (`kappa = -J₂/J₁`)
- `H_z`:   transverse field, scaled by `-h * j` (`h = B/J₁`)

The full Hamiltonian is `H = -J₁ H_nn - J₂ H_nnn - B H_z`.
"""
function annni(kappa, h, n::Int; j=1)
    @assert n >= 4
    mnn  = sum(ntuple(k -> pauli_xx(k - 1, k,     n), n - 1)) # nn terms
    mnnn = sum(ntuple(k -> pauli_xx(k - 1, k + 1, n), n - 2)) # nnn terms
    mz   = sum(ntuple(k -> pauli_z(k - 1, n), n))
    return map(Hermitian, (-j .* mnn, (j * kappa) .* mnnn, (-h * j) .* mz))
end

"""
    krylov_seq(mat_a::AbstractMatrix, v::AbstractVecOrMat, n::Int) -> Matrix

Generate the Krylov sequence `[v, A·v, A²·v, …, Aⁿ⁻¹·v]` for matrix `mat_a`
and starting vector (or block of vectors) `v`.

If `v` is a matrix, each column is treated as an independent starting vector and
the output columns interleave the sequences block-by-block.
"""
function krylov_seq(mat_a::AbstractMatrix, v::AbstractVecOrMat, n::Int)
    step = size(v, 2)   # Krylov block size
    v = repeat(v, outer=(1, n))
    @assert n * step == size(v, 2)
    for j=(step + 1):step:n * step
        v[:, j:end] = mat_a * v[:, j:end]
    end
    return v
end

"""
    krylov_matrix(h::Hermitian; start_v, krylov_size=5, krylov_skip=0,
                  qfd=false, spectrum_scale=1.) -> Matrix

Construct the Krylov matrix from Hamiltonian `h`.

# Keyword arguments
- `start_v`: starting vector or block of vectors.
- `krylov_size`: total number of Krylov steps; output has `krylov_size × block_size` columns.
- `krylov_skip`: discard the first `krylov_skip` blocks from the output.
- `qfd`: if `true`, use the Quantum Filter Diagonalization (QFD) variant, which
  builds the sequence in the eigenbasis via `exp(i λ t)` propagation.
- `spectrum_scale`: scaling factor applied to the eigenvalues in QFD mode
  (ignored when `qfd=false`).
"""
function krylov_matrix(h::Hermitian; start_v::AbstractVecOrMat,
		       krylov_size::Int=5, krylov_skip::Int=0,
		       qfd=false, spectrum_scale=1.)
    @assert krylov_size > krylov_skip

    if qfd
        l, u = eigen(h)
        l_1, l_n = extrema(l)
        #@. l = (l - l_1) / (l_n - l_1) # Rescale/shift eigvals to [0, 1]
	l = spectrum_scale * l ./ maximum(abs, extrema(l))
        start_v = complex(start_v)
        mat_c = u * krylov_seq(diagm(exp.(im * l)), u' * start_v, krylov_size)
    else
        # h is normalized by its norm to prevent eigenvalue blow-up in the power sequence
        mat_c = krylov_seq(h / norm(h), start_v, krylov_size)
    end
    block_sz = trunc(Int, size(mat_c, 2) / krylov_size)
    mat_c = mat_c[:, (1 + block_sz * krylov_skip):end]
    return mat_c
end

# End of module
end;
