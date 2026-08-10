#!/bin/bash
# =============================================================================
# Frontier (OLCF) batch script — replaces submit_script.sh (Polaris/Cobalt).
#
#   sbatch submit_frontier.sh
#
# Frontier is Slurm, not PBS/Cobalt: no $COBALT_NODEFILE, no `mpirun -f`.
# srun reads the allocation directly.
#
# EDIT THIS: -A must be your OLCF project ID (e.g. CFD###). Without it the job
# is rejected at submit time.
# =============================================================================
#SBATCH -A CHANGE_ME_PROJECT_ID
#SBATCH -J LBM3D
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
# --gpu-bind=none: we pin GCDs ourselves in set_affinity_gpu_frontier.sh.
# Slurm's --gpu-bind=closest is documented to hang GPU-aware MPI on Frontier,
# and this code cannot run without GPU-aware MPI.
#SBATCH --gpu-bind=none

cd "$SLURM_SUBMIT_DIR" || exit 1

# Same env the build used — module state does not survive into the job.
source ./env_frontier.sh

RANKS_PER_NODE=8                      # 1 MPI rank per GCD (4 MI250X = 8 GCDs)
NODES=$SLURM_JOB_NUM_NODES
PROCS=$((NODES * RANKS_PER_NODE))

echo "nodes=$NODES  ranks=$PROCS  (${RANKS_PER_NODE}/node, 1 per GCD)"

# -c 7 : 7 cores per rank. Frontier's low-noise mode reserves the first core of
#        each of the 8 L3 regions, so 56 of 64 cores are allocatable -> 56/8 = 7.
#        This also makes local rank i land on L3 region i, which is what the
#        GCD mapping table in set_affinity_gpu_frontier.sh assumes.
srun -N "$NODES" -n "$PROCS" --ntasks-per-node="$RANKS_PER_NODE" \
     -c 7 --gpus-per-node=8 --gpu-bind=none \
     ./set_affinity_gpu_frontier.sh ./LBM3D_frontier
