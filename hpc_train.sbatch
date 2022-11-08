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

module purge
singularity exec --nv --overlay /scratch/projects/yzlab/yw5806/multistep/t5chem.ext3 \
        /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
        /bin/bash -c "source /ext3/env.sh; cd $ROOT/t5chem; python $ROOT/t5chem/run_trainer.py --data_dir $ROOT/data/smiles_mod_train_0 --output_dir $ROOT/out/exp_t5_003 --task_type pretrain --num_epoch 8 --vocab_name simple --add_tokens '<mod>' --add_tokens '</mod>' "

