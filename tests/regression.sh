#!/usr/bin/env bash
#
# tests/regression.sh -- physical regression test for IMEXLBM.
#
#   usage: tests/regression.sh [path-to-imexlbm]      (default ./imexlbm)
#
# What is checked, and why these are the right things to check:
#
#   1. Mass conservation. sum_i f_i is an exact invariant of the discrete
#      dynamics -- BGK collision reproduces it and streaming on a periodic
#      domain is a permutation (see the comment on LBM::Conserved). Any drift
#      beyond rounding is a bug in streaming, in the halo exchange, or a race.
#
#   2. Momentum conservation. Likewise exact, and the Taylor-Green initial
#      field has zero net momentum, so it must stay at zero.
#
#   3. Decomposition independence. Collision is per-cell and streaming is a
#      permutation, so the field is bitwise independent of how the domain is
#      split across ranks; only the order of the final reduction differs.
#      Running the same case on 1, 2 and 4 ranks must therefore agree to near
#      machine epsilon. This is the most valuable check here: the Cartesian
#      decomposition and the 26-neighbour halo exchange are the most intricate
#      part of the code, and the part that is hardest to eyeball on a machine
#      where you get one job per queue wait.
#
# What is deliberately NOT checked: monotone decay of the kinetic energy. The
# distributions are initialised at equilibrium with an unrelaxed pressure field,
# so the first tens of steps carry an acoustic transient and ke genuinely
# oscillates (9.852 -> 10.219 -> 9.889 at 32^3). Asserting monotonicity would be
# a flaky test that is also wrong about the physics.
#
# Tolerances are ~100x the values measured on Kokkos 5.1.1 OpenMP + Open MPI:
#   mass drift within a run      1.2e-14
#   mass across rank counts      1.4e-14
#   kinetic energy across ranks  1.0e-14
# If a tolerance ever has to be loosened, find out why first -- on these
# invariants, a growing residual is evidence, not noise.

set -euo pipefail

TOL_MASS_DRIFT=1e-12   # relative
TOL_MOMENTUM=1e-10     # absolute; the exact initial value is ~1e-14
TOL_RANK_AGREE=1e-12   # relative
RANKS="1 2 4"

# Resolve the executable before changing directory.
raw=${1:-./imexlbm}
EXE="$(cd "$(dirname "$raw")" && pwd)/$(basename "$raw")"
if [ ! -x "$EXE" ]; then
    echo "FAIL  no executable at $EXE" >&2
    exit 1
fi

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

# 32^3 for 20 steps, reporting every step. Small enough to run in seconds on a
# CI runner, large enough that the decomposition is non-trivial at 4 ranks.
printf '1.0  0.1  1600\n0.05  20  1\n32  32  32\n' > input.in

echo "running $EXE on ${RANKS// /, } rank(s)"
for n in $RANKS; do
    if ! OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
         mpirun -n "$n" --oversubscribe "$EXE" > "run.$n.log" 2>&1; then
        echo "FAIL  solver exited non-zero on $n rank(s)" >&2
        tail -20 "run.$n.log" >&2
        exit 1
    fi
    grep '^cons' "run.$n.log" > "cons.$n" || true
    if [ ! -s "cons.$n" ]; then
        echo "FAIL  no 'cons' diagnostic lines on $n rank(s)" >&2
        echo "      the binary predates LBM::Conserved, or the run produced no output" >&2
        exit 1
    fi
    if grep -qiE 'nan|inf' "cons.$n"; then
        echo "FAIL  non-finite value in the diagnostics on $n rank(s)" >&2
        grep -iE 'nan|inf' "cons.$n" | head -3 >&2
        exit 1
    fi
done
echo

status=0

# --- 1 and 2: invariants hold within each run ------------------------------
# Field layout of a 'cons' line under FS=[ |]+ :
#   1 cons  2 step  3 <it>  4 mass  5 <mass>  6 px  7 <px>
#   8 py  9 <py>  10 pz  11 <pz>  12 ke  13 <ke>
for n in $RANKS; do
    awk -F'[ |]+' -v n="$n" -v tm="$TOL_MASS_DRIFT" -v tp="$TOL_MOMENTUM" '
        NR == 1 { m0 = $5 }
        {
            d = ($5 - m0) / m0; if (d < 0) d = -d; if (d > dm) dm = d
            for (i = 7; i <= 11; i += 2) { p = $i; if (p < 0) p = -p; if (p > dp) dp = p }
        }
        END {
            ok = 1
            if (dm > tm) { printf "FAIL  mass not conserved on %s rank(s): max relative drift %.3e > %s\n", n, dm, tm; ok = 0 }
            else         { printf "ok    mass conserved on %s rank(s)              (max drift %.3e)\n", n, dm }
            if (dp > tp) { printf "FAIL  momentum not conserved on %s rank(s): max |p| %.3e > %s\n", n, dp, tp; ok = 0 }
            else         { printf "ok    net momentum stays zero on %s rank(s)     (max |p| %.3e)\n", n, dp }
            exit ok ? 0 : 1
        }' "cons.$n" || status=1
done
echo

# --- 3: the answer does not depend on the decomposition --------------------
ref=${RANKS%% *}
for n in $RANKS; do
    [ "$n" = "$ref" ] && continue
    paste "cons.$ref" "cons.$n" | awk -F'[ |\t]+' -v n="$n" -v r="$ref" -v t="$TOL_RANK_AGREE" '
        {
            m1 = $5; k1 = $13; m2 = $18; k2 = $26
            d = (m2 - m1) / m1; if (d < 0) d = -d; if (d > dm) dm = d
            d = (k2 - k1) / k1; if (d < 0) d = -d; if (d > dk) dk = d
        }
        END {
            ok = 1
            if (dm > t) { printf "FAIL  mass differs between %s and %s ranks: %.3e > %s\n", r, n, dm, t; ok = 0 }
            else        { printf "ok    mass agrees between %s and %s ranks           (%.3e)\n", r, n, dm }
            if (dk > t) { printf "FAIL  kinetic energy differs between %s and %s ranks: %.3e > %s\n", r, n, dk, t; ok = 0 }
            else        { printf "ok    kinetic energy agrees between %s and %s ranks (%.3e)\n", r, n, dk }
            exit ok ? 0 : 1
        }' || status=1
done

echo
if [ "$status" -ne 0 ]; then
    echo "regression test FAILED"
    exit 1
fi
echo "regression test passed"
