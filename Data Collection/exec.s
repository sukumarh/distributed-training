#!/bin/bash
#SBATCH --cpus-per-task=28
#SBATCH --reservation=chung
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:4
#SBATCH --mem=102400
#SBATCH --job-name=training_logger
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

#module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2 

#source ~/pytorch_env/py3.6.3/bin/activate


python distributed_trainer.py -c "47,46,45,44" -w 16 -d data/ -s training_data/


