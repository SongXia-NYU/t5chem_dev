#!/bin/bash

#SBATCH --job-name=t5chem_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00
#SBATCH --mem=100GB

CONFIG_FILE=$1

module purge
singularity exec --nv --overlay $SCRATCH/t5chem_dev_song/t5chem_dev/t5chem_dev.ext3:ro \
	/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
        /bin/bash -c "source /ext3/env.sh; python -m torch.distributed.launch --use_env --nproc_per_node=3 --master_port=$RANDOM t5chem/run_trainer.py --config_file $CONFIG_FILE  "

