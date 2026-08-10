#!/bin/bash
# =============================================================================
# Frontier strong-scaling sweep: 1 -> 256 nodes (8 -> 2048 ranks).
#
#   make -f Makefile.frontier SCALING=1   # build LBM3D_frontier_scaling FIRST
#   sbatch submit_strong_scaling.sh
#   ./parse_scaling.sh scaling_<jobid>    # after it finishes
#
# WHY ONE JOB INSTEAD OF NINE
# ---------------------------
# Nine separate sbatch jobs would queue independently, land on different nodes,
# and take days of wall-clock to all schedule. This asks for 256 nodes once and
# runs every size inside that allocation.
#
# Slurm makes the node subsetting easier than PBS did: srun -N <n> simply uses
# the first n nodes of the allocation, so no hostfile juggling is needed.
#
# A side benefit: every point runs on the same physical nodes, so node-to-node
# variation cannot masquerade as a scaling effect.
#
# EDIT THIS: -A must be your OLCF project ID.
# =============================================================================
#SBATCH -A CHANGE_ME_PROJECT_ID
#SBATCH -J LBM_strong_scaling
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 256
#SBATCH --gpu-bind=none

set -u

cd "$SLURM_SUBMIT_DIR" || exit 1
source ./env_frontier.sh

# ---- study parameters -------------------------------------------------------
# Strong scaling: the GLOBAL problem size is FIXED; only the node count varies.
#
# 768^3, same as Aurora, so the two are directly comparable:
#   768^3 x ~696 B/cell = 315 GB over 8 GCDs = ~39 GB/GCD against 64 GB HBM.
# 768 = 2^8 x 3 divides cleanly across every rank count in the sweep (8 x 2^k),
# and at 256 nodes each rank still holds 768^3/2048 = 221k cells.
#
# NOTE Polaris uses 512^3 instead — its 40 GB A100s cannot hold 768^3 on one
# node. Only Frontier and Aurora numbers are directly comparable as shipped.
NX=${NX:-768}
NY=${NY:-768}
NZ=${NZ:-768}

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
RANKS_PER_NODE=8          # one rank per GCD (4 MI250X = 8 GCDs)

EXE=LBM3D_frontier_scaling

# ---- staging ----------------------------------------------------------------
RUNDIR="${SLURM_SUBMIT_DIR}/scaling_${SLURM_JOB_ID}"
mkdir -p "$RUNDIR" || exit 1
cd "$RUNDIR" || exit 1

if [ ! -x "${SLURM_SUBMIT_DIR}/${EXE}" ]; then
    echo "FATAL: ${EXE} not found in ${SLURM_SUBMIT_DIR}"
    echo "       build it first:  make -f Makefile.frontier SCALING=1"
    exit 1
fi
cp "${SLURM_SUBMIT_DIR}/${EXE}" .
cp "${SLURM_SUBMIT_DIR}/set_affinity_gpu_frontier.sh" .
chmod +x set_affinity_gpu_frontier.sh

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
MACHINE="Frontier (OLCF, MI250X)"
RANKS_PER_NODE=${RANKS_PER_NODE}
GRID="${NX}x${NY}x${NZ}"
TIME_STEPS=${TIME_STEPS}
EOF

AVAIL_NODES=$SLURM_JOB_NUM_NODES
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

    NRANKS=$((N * RANKS_PER_NODE))
    echo "==> ${N} node(s), ${NRANKS} ranks  ($(date '+%H:%M:%S'))"

    # -c 7: low-noise mode reserves the first core of each of the 8 L3 regions,
    # so 56 of 64 cores are allocatable -> 56/8 = 7. This also puts local rank i
    # on L3 region i, which the GCD mapping in set_affinity_gpu_frontier.sh
    # assumes. --gpu-bind=none because Slurm's =closest hangs GPU-aware MPI.
    srun -N "$N" -n "$NRANKS" --ntasks-per-node="$RANKS_PER_NODE" \
         -c 7 --gpus-per-node=8 --gpu-bind=none \
         ./set_affinity_gpu_frontier.sh "./${EXE}" \
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
