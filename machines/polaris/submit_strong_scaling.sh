#!/bin/bash -l
# =============================================================================
# Polaris strong-scaling sweep: 1 -> 256 nodes (4 -> 1024 ranks).
#
#   make -f Makefile.polaris SCALING=1    # build LBM3D_polaris_scaling FIRST
#   qsub submit_strong_scaling.sh
#   ./parse_scaling.sh scaling_<jobid>    # after it finishes
#
# WHY ONE JOB INSTEAD OF NINE
# ---------------------------
# One qsub per node count fights the queue policy: debug is 1-2 nodes and
# debug-scaling is 1-10, so most of the sweep cannot even be submitted there,
# and you would be babysitting nine jobs.
#
# Instead this asks for 256 nodes once (prod, which is 10-496 nodes) and runs
# every size inside that allocation, handing mpiexec a --hostfile that is the
# first N lines of $PBS_NODEFILE. Strong scaling means the work shrinks as N
# grows, so the sweep is dominated by the 1-node run.
#
# A side benefit: every point runs on the same physical nodes, so node-to-node
# variation cannot masquerade as a scaling effect.
#
# EDIT THIS: -A must be your ALCF project name.
# =============================================================================
#PBS -A CHANGE_ME_PROJECT_NAME
#PBS -N LBM_strong_scaling
#PBS -l select=256:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q prod
#PBS -k doe

set -u

cd "${PBS_O_WORKDIR}" || exit 1
source ./env.sh

# ---- study parameters -------------------------------------------------------
# Strong scaling: the GLOBAL problem size is FIXED; only the node count varies.
#
# 512^3 here, NOT the 768^3 used on Frontier and Aurora. Polaris' A100s have
# 40 GB each, so a 1-node baseline is the binding constraint:
#   512^3 x ~696 B/cell = 93 GB over 4 GPUs = ~23 GB/GPU   -> fits
#   768^3                = 315 GB over 4 GPUs = ~79 GB/GPU -> does NOT fit
# 512 = 2^9 divides cleanly across every rank count in the sweep (4 x 2^k),
# and at 256 nodes each rank still holds 512^3/1024 = 131k cells.
#
# FOR CROSS-MACHINE COMPARISON you need the same grid on all three. Run
#   qsub -v NX=768,NY=768,NZ=768,NODE_LIST="4 8 16 32 64 128 256" ...
# — 768^3 needs at least 4 Polaris nodes (~20 GB/GPU), so the 1- and 2-node
# points simply do not exist here. Comparing 512^3 numbers against the other
# machines' 768^3 numbers is meaningless.
NX=${NX:-512}
NY=${NY:-512}
NZ=${NZ:-512}

# 2000 steps / 200-step interval = 10 timed intervals. The parser discards the
# first (warm-up) and averages the rest.
TIME_STEPS=${TIME_STEPS:-2000}
INTER=${INTER:-200}

# Physics knobs carried over from input.in. They do not affect throughput --
# System.cpp derives tau from sx, so tau differs from your production runs.
RHO0=1.0
R=0.1
RE=1600
U0=0.05

NODE_LIST=${NODE_LIST:-"1 2 4 8 16 32 64 128 256"}
RANKS_PER_NODE=4          # one rank per A100
NDEPTH=8                  # 32 cores / 4 ranks

EXE=LBM3D_polaris_scaling

# ---- staging ----------------------------------------------------------------
RUNDIR="${PBS_O_WORKDIR}/scaling_${PBS_JOBID%%.*}"
mkdir -p "$RUNDIR" || exit 1
cd "$RUNDIR" || exit 1

if [ ! -x "${PBS_O_WORKDIR}/${EXE}" ]; then
    echo "FATAL: ${EXE} not found in ${PBS_O_WORKDIR}"
    echo "       build it first:  make -f Makefile.polaris SCALING=1"
    exit 1
fi
cp "${PBS_O_WORKDIR}/${EXE}" .
cp "${PBS_O_WORKDIR}/set_affinity_gpu_polaris.sh" .
chmod +x set_affinity_gpu_polaris.sh

# System.cpp reads a hardcoded "input.in" from the CWD, so write it here.
cat > input.in <<EOF
${RHO0}  ${R}  ${RE}
${U0}  ${TIME_STEPS}  ${INTER}
${NX}  ${NY}  ${NZ}
=============
rho R Re
u0 Time inter
nx ny nz
EOF

# Describes the sweep for parse_scaling.sh, so that script is machine-agnostic.
cat > sweep.meta <<EOF
MACHINE="Polaris (ALCF, A100 40GB)"
RANKS_PER_NODE=${RANKS_PER_NODE}
GRID="${NX}x${NY}x${NZ}"
TIME_STEPS=${TIME_STEPS}
EOF

AVAIL_NODES=$(wc -l < "$PBS_NODEFILE")
echo "==> allocation has ${AVAIL_NODES} nodes"
echo "==> global grid ${NX}x${NY}x${NZ}, ${TIME_STEPS} steps, interval ${INTER}"
echo "==> results in ${RUNDIR}"
echo

# ---- sweep ------------------------------------------------------------------
for N in $NODE_LIST; do
    if [ "$N" -gt "$AVAIL_NODES" ]; then
        echo "==> SKIP ${N} nodes (allocation only has ${AVAIL_NODES})"
        continue
    fi

    head -n "$N" "$PBS_NODEFILE" > "hostfile.${N}"
    NRANKS=$((N * RANKS_PER_NODE))

    echo "==> ${N} node(s), ${NRANKS} ranks  ($(date '+%H:%M:%S'))"

    mpiexec -n "$NRANKS" --ppn "$RANKS_PER_NODE" \
            --depth="$NDEPTH" --cpu-bind depth \
            --hostfile "hostfile.${N}" \
            ./set_affinity_gpu_polaris.sh "./${EXE}" \
            > "run_N${N}.log" 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "    FAILED (exit ${rc}) -- see run_N${N}.log"
    else
        echo "    ok: $(grep -c MLUPS "run_N${N}.log") interval line(s)"
    fi
done

echo
echo "==> sweep complete. Summarise with:"
echo "    ./parse_scaling.sh ${RUNDIR}"
