# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#!/bin/sh
# Generate the plots for the paper

DATA_PATH="data"
FAILED=0

merge_fig_hori(){
    for f in "$1" "$2"; do
        if [ ! -f "$f" ]; then
            echo "merge_fig_hori: input file '$f' not found, skipping merge into $3.pdf" >&2
            return 1
        fi
    done

echo "\\\documentclass[border=1mm]{standalone}
\\\usepackage{tikz}
\\\usetikzlibrary {positioning}

\\\begin{document}
\\\begin{tikzpicture}
    \\\node (A) {\\\includegraphics[]{$1}};
    \\\node (B) [right=of A] {\\\includegraphics{$2}};

    \\\node[anchor=north] at (A.south) {\\\Large\\\textbf{(a)}};
    \\\node[anchor=north] at (B.south) {\\\Large\\\textbf{(b)}};
\\\end{tikzpicture}
\\\end{document}" | pdflatex -interaction=batchmode -halt-on-error -jobname=$3 >/dev/null
    status=$?
    if [ $status -ne 0 ]; then
        echo "merge_fig_hori: pdflatex failed for $3, see $3.log" >&2
    fi
    return $status
}

julia eigvals-plot.jl $DATA_PATH/new-LGTZ2-blocked-q64-k80/time_r-overlaps-LGTZ2-ghzb2-q64-k80.jld2 --xlabel "\mu=g" \
    --eigvals_thr "1e-6" \
    --data-labels "KTR \rightarrow KQD" --ref-labels "KTR \rightarrow DMRG" \
    --ref-energy $DATA_PATH/new-LGTZ2-blocked-q64-k80/time_r-overlaps-LGTZ2-q64-dmrg.jld2 \
    --output time_r-LGTZ2-blocked-lambda0-rel_err.pdf --legend bottomright \
    || FAILED=1

julia eigvals-plot.jl $DATA_PATH/Ising-blocked-q64-k128/time_r-overlaps-Ising-ghzb*-q64-k128.jld2 --xlabel "\gamma" \
    --eigvals_thr "1e-6" \
    --data-labels "s=2" "s=8" \
    --ref-energy $DATA_PATH/Ising-blocked-q64-k128/time_r-overlaps-Ising-q64-dmrg.jld2 \
    --output time_r-Ising-blocked-lambda0-rel_err.pdf --legend bottomright \
    || FAILED=1

julia eigvals-plot.jl $DATA_PATH/time_r-overlaps-Ising-implicit_hadamard-q64-k128.jld2 --xlabel "\gamma" \
    --eigvals_thr "1e-6" \
    --data-labels "KTR \rightarrow KQD" --ref-labels "KTR \rightarrow DMRG" \
    --ref-energy $DATA_PATH/Ising-blocked-q64-k128/time_r-overlaps-Ising-q64-dmrg.jld2 \
    --output time_r-Ising-implicit_hadamard-lambda0-rel_err.pdf --legend bottomright \
    || FAILED=1

# Aggregate plots: time_r-Ising-impl_hadamard.pdf, time_r-Ising-implicit_hadamard-lambda0-rel_err.pdf
merge_fig_hori time_r-Ising-impl_hadamard.pdf time_r-Ising-implicit_hadamard-lambda0-rel_err.pdf time_r-Ising-impl_hadamard-aggregated \
    || FAILED=1

# *** Implicit Hadamard extended ***
implhext() {
julia eigvals-plot.jl \
    $DATA_PATH/Ising-blocked-q64-k128-implicit_hadamard_ext/time_r-overlaps-Ising-q64-k128-implicit_hadamard_ext-projte1.jld2 \
    $DATA_PATH/Ising-blocked-q64-k128-implicit_hadamard_ext/time_r-overlaps-Ising-q64-k128-implicit_hadamard_ext-projte16.jld2 \
    --xlabel "\gamma" \
    --eigvals_thr "1e-6" \
    --data-labels "1" "2^s" \
    --legend bottomright "$@"
}

IMPLHEXT_PREFIX="time_r-Ising-implicit_hadamard_extended-lambda0-rel_err"
implhext --output $IMPLHEXT_PREFIX-kqd.pdf \
    || FAILED=1
implhext --output $IMPLHEXT_PREFIX-dmrg.pdf --ref-plot-only \
    --ref-energy $DATA_PATH/Ising-blocked-q64-k128/time_r-overlaps-Ising-q64-dmrg.jld2 \
    || FAILED=1
merge_fig_hori $IMPLHEXT_PREFIX-kqd.pdf $IMPLHEXT_PREFIX-dmrg.pdf $IMPLHEXT_PREFIX \
    || FAILED=1

# Cleanup LaTex processor intermediate files
rm -f *.log *.aux

if [ "$FAILED" -ne 0 ]; then
    echo "time_r-run_plots.sh: one or more jobs failed, see messages above" >&2
    exit 1
fi

