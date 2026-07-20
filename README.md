# Krylov Time Reversal (KTR)

Reference implementation for the paper:

> N. Mariella, E. Rico, A. Byrne, S. Zhuk, *"Quantum Krylov Subspace Diagonalization via Time Reversal Symmetries"*, [arXiv:2507.22559](https://arxiv.org/abs/2507.22559).

Krylov Time Reversal (KTR) is a protocol for quantum Krylov subspace diagonalization that avoids controlled unitaries (e.g. the Hadamard test) by exploiting a time-reversal symmetry of the Hamiltonian dynamics. Krylov matrix elements are instead recovered as the real/imaginary parts of overlaps built from time-reversal-symmetric initial states, which lowers circuit depth and total evolution time compared to standard quantum Krylov diagonalization (QKD). The method is validated here on the transverse-field Ising chain and on a Z₂ lattice gauge theory, using classical MPS/MPO simulation (via [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl)) as a stand-in for the quantum evolution.

## Code layout

Core library:
- **`KTR.jl`** — dense-matrix building blocks: Pauli operators, the Ising and ANNNI Hamiltonians, Krylov sequence/matrix construction (including the QFD variant), and linear-algebra helpers (spectral thresholding, PSD projection, Hermitian/Toeplitz matrix construction from a single row).
- **`KTRMPS.jl`** — the MPS/MPO counterpart, built on `KTR.jl`: Hamiltonian and time-reversal MPOs for the Ising chain and the Z₂ lattice gauge theory (LGT), Suzuki–Trotter time evolution, block-Krylov sequence generation over MPS, blocked-GHZ initial state construction, and the overlap-matrix routines (`overlaps_time_r_with_sign`, `overlap_mps`, ...) used to assemble the generalized eigenvalue problem for both KTR and standard QKD.

Drivers and recipes:
- **`eigvals.jl`** — main driver. Scans a range of Hamiltonian parameters, builds the KTR and reference-QKD overlap matrices at each point, and saves everything to a `.jld2` file. It works by `include`-ing exactly one *recipe* file (edit the `include` line at the top to switch configuration):
  - `eigvals-recipe_ising-block.jl` — transverse-field Ising chain, blocked-GHZ initial states (`v0`/`v0_perp`).
  - `eigvals-recipe_ising-implicit_hadamard.jl` — transverse-field Ising chain, single-reference implicit-Hadamard-test construction.
  - `eigvals-recipe_lgt-block.jl` — Z₂ LGT chain, blocked-GHZ initial state with gauge-sector projection/penalty.
- **`eigvals-dmrg.jl`** — computes reference ground-state energies with DMRG for the same recipe/parameter scan, used as ground truth when plotting.
- **`eigvals-plot.jl`** — command-line tool (`ArgParse`-based) that reads one or more `.jld2` overlap files, solves the generalized eigenvalue problem (with spectral thresholding on the Gram matrix), and plots the relative error of the extracted ground-state energy against the scan parameter; optionally overlays a DMRG reference curve.
- **`generate_plots.sh`** — regenerates the figures used in the paper by invoking `eigvals-plot.jl` against precomputed data expected under `data/` (see below), then merges pairs of figures side by side with a small LaTeX/TikZ snippet (requires `pdflatex`).

Standalone checks:
- **`test-krylov-time-reversal.jl`** — small dense (exact-diagonalization) sanity check of the KTR method against direct diagonalization of the Ising Hamiltonian.
- **`implicit_hadamard-ising.jl`** / **`implicit_hadamard-ising-extended.jl`** — validate the implicit-Hadamard-test construction (single-reference and multi-term/projected variants) against direct QKD overlaps.
- **`testing.jl`** — unit tests (`Test` stdlib) for `KTR.jl`/`KTRMPS.jl`: Hamiltonian/MPO consistency, Trotter evolution accuracy, blocked-GHZ state construction.

## Requirements

Julia (tested with 1.12) plus the following packages: `ITensorMPS`, `Infinities`, `JLD2`, `Plots`, `LaTeXStrings`, `ArgParse`, `ProgressBars`. `LinearAlgebra`, `Logging`, `Random`, and `Test` are part of the Julia standard library.

```julia
using Pkg
Pkg.add(["ITensorMPS", "Infinities", "JLD2", "Plots", "LaTeXStrings", "ArgParse", "ProgressBars"])
```

`generate_plots.sh` additionally needs a `pdflatex` installation (with the `tikz` package) to merge figures side by side.

## Usage

Run the unit tests:

```sh
julia testing.jl
```

Generate KTR/QKD overlap data for a given configuration (pick the recipe by editing the `include` in `eigvals.jl`, then run):

```sh
julia eigvals.jl
```

this writes a `time_r-overlaps-<model_name>-q<n>-k<krylov_size>.jld2` file with both the KTR and QKD overlap matrices for every scanned parameter value.

Generate the matching DMRG reference energies (same recipe):

```sh
julia eigvals-dmrg.jl
```

Plot the relative error of the extracted ground energy against a scan parameter:

```sh
julia eigvals-plot.jl path/to/overlaps.jld2 --xlabel "\gamma" --ref-energy path/to/dmrg.jld2 --output plot.pdf
```

Regenerate all paper figures at once (expects the corresponding `.jld2` datasets to already exist under `data/`):

```sh
./generate_plots.sh
```

## Data

`data/` is where the `.jld2` overlap datasets consumed by `eigvals-plot.jl` and `generate_plots.sh` are expected to live, organized in one subfolder per configuration/system size. These files are generated locally with `eigvals.jl`/`eigvals-dmrg.jl` and are not tracked in git (see `.gitignore`).

## Citation

If you use this code, please cite:

```bibtex
@article{mariella2025ktr,
  title   = {Quantum Krylov Subspace Diagonalization via Time Reversal Symmetries},
  author  = {Mariella, Nicola and Rico, Enrique and Byrne, Adam and Zhuk, Sergiy},
  journal = {arXiv preprint arXiv:2507.22559},
  year    = {2025}
}
```
