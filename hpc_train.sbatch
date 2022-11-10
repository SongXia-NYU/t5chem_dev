#!/bin/bash

#SBATCH --job-name=t5chem_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

ROOT=$SCRATCH/scripts/t5chem_dev
CONFIG_FILE=$1

module purge
singularity exec --nv --overlay /home/sx801/conda_envs/t5chem.sqf \
        /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
        /bin/bash -c "source /ext3/env.sh; export PYTHONPATH=$ROOT:$PYTHONPATH; python $ROOT/t5chem/run_trainer.py --config_file $CONFIG_FILE  "

