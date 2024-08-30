#!/bin/bash
#SBATCH --job-name=v2midi_train
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/v2midi_train.out
#SBATCH --error=outputs/v2midi_train.err
#SBATCH -A fkc@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=19:00:00

module purge
export PYTHONUSERBASE=$WORK/.local_automatic1111

# Load modules
module load git
module load pytorch-gpu/py3/2.1.1

# Launch the script
python train.py
