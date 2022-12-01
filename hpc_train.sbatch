#!/bin/bash

#SBATCH --job-name=t5chem_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

CONFIG_FILE=$1

module purge
singularity exec --nv --overlay $SCRATCH/t5chem_dev_song/t5chem_dev/t5chem_dev.ext3 \
	/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
        /bin/bash -c "python t5chem/run_trainer.py --config_file $CONFIG_FILE  "

