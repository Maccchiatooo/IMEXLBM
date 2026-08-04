#!/bin/bash -l
# =============================================================================
# Aurora strong-scaling sweep: 1 -> 256 nodes (12 -> 3072 ranks).
#
#   make -f Makefile.aurora SCALING=1     # build LBM3D_aurora_scaling FIRST
#   qsub submit_strong_scaling.sh
#   ./parse_scaling.sh scaling_<jobid>    # after it finishes
#
# WHY ONE JOB INSTEAD OF NINE
# ---------------------------
# The obvious approach — one qsub per node count — does not work well here:
# debug-scaling and debug each allow only ONE job running/accruing/queued per
# user, so you would have to babysit nine submissions serially.
#
# Instead this asks for 256 nodes once and runs every size inside that single
# allocation, handing mpiexec a --hostfile that is the first N lines of
# $PBS_NODEFILE. Same nodes, same job, no queue ping-pong. Strong scaling means
# the work shrinks as N grows, so the whole sweep is dominated by the 1-node
# run and fits comfortably in the 1-hour debug-scaling limit.
#
# A side benefit: every point runs on the same physical nodes, so node-to-node
# performance variation cannot masquerade as a scaling effect.
#
# EDIT THIS: -A must be your ALCF project name.
# =============================================================================
#PBS -A CHANGE_ME_PROJECT_NAME
#PBS -N LBM_strong_scaling
#PBS -l select=256
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -k doe

set -u

cd "${PBS_O_WORKDIR}" || exit 1
source ./env_aurora.sh

# ---- study parameters -------------------------------------------------------
# Strong scaling: the GLOBAL problem size is FIXED; only the node count varies.
#
# 768^3 is chosen so that:
#   * it fits on 1 node -- 768^3 x ~696 B/cell = 315 GB over 12 tiles
#     = ~26 GB/tile against 64 GB of HBM per tile, leaving room for the ~38%
#     ghost-layer overhead that appears at the 256-node end;
#   * 768 = 2^8 x 3 divides cleanly across every rank count in the sweep
#     (12 x 2^k for k = 0..8), so no point is penalised by load imbalance;
#   * at 256 nodes each rank still holds ~48x48x64 = 147k cells, small but not
#     degenerate.
NX=${NX:-768}
NY=${NY:-768}
NZ=${NZ:-768}

# 2000 steps / 200-step reporting interval = 10 timed intervals per run. The
# parser discards the first (warm-up: SYCL kernel load, first-touch allocation,
# MPI connection setup) and averages the rest.
TIME_STEPS=${TIME_STEPS:-2000}
INTER=${INTER:-200}

# Physics knobs are carried over from your input.in. They do not affect
# throughput -- note that System.cpp derives tau from sx, so tau differs from
# your 256^3 production runs. That is fine for a performance study.
RHO0=1.0
R=0.1
RE=1600
U0=0.05

NODE_LIST=${NODE_LIST:-"1 2 4 8 16 32 64 128 256"}
RANKS_PER_NODE=12

EXE=LBM3D_aurora_scaling

# ---- staging ----------------------------------------------------------------
# Run in a per-job directory so the sweep never clobbers your own input.in,
# and so results are reproducible and self-describing.
RUNDIR="${PBS_O_WORKDIR}/scaling_${PBS_JOBID%%.*}"
mkdir -p "$RUNDIR" || exit 1
cd "$RUNDIR" || exit 1

if [ ! -x "${PBS_O_WORKDIR}/${EXE}" ]; then
    echo "FATAL: ${EXE} not found in ${PBS_O_WORKDIR}"
    echo "       build it first:  make -f Makefile.aurora SCALING=1"
    exit 1
fi
cp "${PBS_O_WORKDIR}/${EXE}" .
cp "${PBS_O_WORKDIR}/set_affinity_gpu_aurora.sh" .
chmod +x set_affinity_gpu_aurora.sh

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
MACHINE="Aurora (ALCF, PVC Max 1550)"
RANKS_PER_NODE=${RANKS_PER_NODE}
GRID="${NX}x${NY}x${NZ}"
TIME_STEPS=${TIME_STEPS}
EOF

# See submit_aurora.sh for why this is an explicit list and not --cpu-bind depth
# (cores 0 and 52 are reserved for system services as of 2025-03-31).
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"

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

    mpiexec -n "$NRANKS" -ppn "$RANKS_PER_NODE" \
            --hostfile "hostfile.${N}" \
            --cpu-bind "$CPU_BIND" \
            ./set_affinity_gpu_aurora.sh "./${EXE}" \
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
