#!/bin/bash
# =============================================================================
# Summarise a strong-scaling sweep produced by submit_strong_scaling.sh.
#
#   ./parse_scaling.sh scaling_<jobid>
#   ./parse_scaling.sh scaling_<jobid> --csv > scaling.csv
#
# Machine-agnostic: submit_strong_scaling.sh writes a sweep.meta into the run
# directory (ranks/node, grid, machine), so this exact script works unchanged on
# Polaris (4 ranks/node), Frontier (8) and Aurora (12).
#
# Methodology:
#   * main.cpp prints one "... | %10.2f MLUPS" line every `inter` steps, using
#     the GLOBAL cell count -- the right normalisation for strong scaling
#     (fixed total work, so MLUPS should rise linearly with nodes).
#   * The FIRST interval of each run is DISCARDED: it absorbs kernel load,
#     first-touch allocation and MPI connection setup. Averaging it in would
#     penalise the short high-node-count runs far more than the 1-node run and
#     manufacture a scaling cliff that is not real.
#   * min/max are printed so you can see run-to-run spread rather than trusting
#     a bare mean.
# =============================================================================
set -u

DIR=${1:-.}
CSV=0
[ "${2:-}" = "--csv" ] && CSV=1

if [ ! -d "$DIR" ]; then
    echo "usage: $0 <scaling-run-directory> [--csv]" >&2
    exit 1
fi

logs=$(ls "$DIR"/run_N*.log 2>/dev/null | sort -t N -k2 -n)
if [ -z "$logs" ]; then
    echo "no run_N*.log files found in $DIR" >&2
    exit 1
fi

RANKS_PER_NODE=12
GRID="?"
MACHINE="?"
if [ -f "$DIR/sweep.meta" ]; then
    # shellcheck disable=SC1090
    . "$DIR/sweep.meta"
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

for log in $logs; do
    n=$(basename "$log" .log); n=${n#run_N}

    read -r mean mn mx cnt <<EOF
$(awk '/MLUPS/ {
           v = ""
           for (i = 1; i <= NF; i++) if ($i == "MLUPS") v = $(i-1)
           if (v !~ /^[0-9]+(\.[0-9]+)?$/ || v+0 <= 0) next   # not a real datum
           k++
           if (k == 1) next                 # discard warm-up interval
           s += v; c++
           if (mn == "" || v < mn) mn = v
           if (mx == "" || v > mx) mx = v
       }
       END { if (c > 0) printf "%.2f %.2f %.2f %d\n", s/c, mn, mx, c
             else       printf "0 0 0 0\n" }' "$log")
EOF

    if [ "$cnt" -eq 0 ]; then
        echo "warn: $log has no usable MLUPS intervals (run failed, or Time <= inter)" >&2
    fi
    echo "$n $mean $mn $mx $cnt" >> "$tmp"
done

sort -n "$tmp" -o "$tmp"

base=$(awk 'NR==1 && $1==1 {print $2}' "$tmp")
if [ -z "${base:-}" ] || [ "$(echo "$base" | cut -d. -f1)" = "0" ]; then
    echo "warn: no valid 1-node baseline; speedup/efficiency omitted" >&2
    base=""
fi

if [ "$CSV" -eq 1 ]; then
    echo "nodes,ranks,mlups_mean,mlups_min,mlups_max,intervals,speedup,efficiency_pct"
    awk -v base="$base" -v rpn="$RANKS_PER_NODE" '{
        sp = (base != "" && base > 0) ? $2/base : ""
        ef = (sp != "") ? 100*sp/$1 : ""
        printf "%d,%d,%.2f,%.2f,%.2f,%d,%s,%s\n", $1, $1*rpn, $2, $3, $4, $5,
               (sp==""?"":sprintf("%.2f",sp)), (ef==""?"":sprintf("%.1f",ef))
    }' "$tmp"
    exit 0
fi

echo
echo "Strong scaling — global grid fixed, node count varying"
echo "machine: $MACHINE   grid: $GRID   ranks/node: $RANKS_PER_NODE"
echo "MLUPS = million lattice updates/s over the whole domain (higher is better)"
echo "First reporting interval of each run discarded as warm-up"
echo
printf "%6s %7s %12s %12s %12s %6s %9s %8s\n" \
       "nodes" "ranks" "MLUPS" "min" "max" "n" "speedup" "eff %"
printf "%6s %7s %12s %12s %12s %6s %9s %8s\n" \
       "------" "-------" "------------" "------------" "------------" "------" "---------" "--------"

awk -v base="$base" -v rpn="$RANKS_PER_NODE" '{
    if (base != "" && base > 0) {
        sp = $2/base; ef = 100*sp/$1
        printf "%6d %7d %12.2f %12.2f %12.2f %6d %8.2fx %7.1f\n", $1, $1*rpn, $2, $3, $4, $5, sp, ef
    } else {
        printf "%6d %7d %12.2f %12.2f %12.2f %6d %9s %8s\n", $1, $1*rpn, $2, $3, $4, $5, "-", "-"
    }
}' "$tmp"

echo
echo "Ideal strong scaling = speedup equal to the node count (eff 100%)."
echo "Expect real falloff at the high end: with ghost=3 the halo/interior ratio"
echo "grows as the subdomain shrinks, so communication comes to dominate. That is"
echo "a property of the decomposition, not a measurement artefact."
