#!/bin/bash

#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --job-name="peract square & can v2"
#SBATCH --nodelist="iliad9"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /iliad/u/priyasun/miniconda3/bin/activate
cd /iliad/u/priyasun/interface_testing
source set_env.sh
python scripts/train_peract.py --config cfgs/peract/square.yaml &
python scripts/train_peract.py --config cfgs/peract/can.yaml
wait
