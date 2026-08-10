#!/bin/bash -l
# =============================================================================
# Aurora (ALCF) batch script — replaces submit_script.sh (Polaris/Cobalt).
#
#   qsub submit_aurora.sh
#
# Aurora is PBS Pro + Cray PALS mpiexec. Closer to Polaris than Frontier was,
# but there is no $COBALT_NODEFILE — use $PBS_NODEFILE.
#
# EDIT THIS: -A must be your ALCF project name. Also check -q against the queue
# table: debug is 1-2 nodes / max 1 hr; debug-scaling is 2-256 nodes; prod is
# the routing queue and needs >= 256 nodes; capacity is 1-16 nodes up to 7 days.
#
# IMPORTANT: submit from your project directory (/lus/flare/<project>/...), NOT
# from $HOME and NOT from /soft/modulefiles — ALCF documents that jobs
# submitted from those locations can end abruptly.
# =============================================================================
#PBS -A CHANGE_ME_PROJECT_NAME
#PBS -N LBM3D
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe

cd "${PBS_O_WORKDIR}" || exit 1

# Same env the build used — module state does not survive into the job.
source ./env_aurora.sh

NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS=12                          # 1 MPI rank per GPU tile (6 PVC x 2 tiles)
NTOTRANKS=$((NNODES * NRANKS))

echo "nodes=$NNODES  ranks=$NTOTRANKS  (${NRANKS}/node, 1 per tile)"

# Explicit core list rather than --cpu-bind=depth. Aurora has 104 physical
# cores (2 sockets x 52), but since 2025-03-31 cores 0 and 52 — the first core
# of each socket — are reserved for system services. A plain --depth=8
# round-robin would hand core 0 to rank 0 and collide with those services.
#
# 6 ranks per socket x 8 cores: socket 0 gets 1-48, socket 1 gets 53-100.
# Cores 49-51 and 101-103 are deliberately left idle — 51 usable cores per
# socket does not divide evenly by 6, and an uneven split would misalign the
# rank-to-socket mapping that set_affinity_gpu_aurora.sh depends on.
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"

mpiexec -n "$NTOTRANKS" -ppn "$NRANKS" \
        --cpu-bind "$CPU_BIND" \
        ./set_affinity_gpu_aurora.sh ./LBM3D_aurora
