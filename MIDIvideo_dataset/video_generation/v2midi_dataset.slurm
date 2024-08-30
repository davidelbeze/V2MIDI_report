#!/bin/bash
#SBATCH --job-name=v2midi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -A kcr@v100
#SBATCH --array=0-19%20
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=16:00:00

module purge
export PYTHONUSERBASE=$WORK/.local_automatic1111

# Load modules
module load git
module load pytorch-gpu/py3/2.1.1

# Launch the script
python main_video_generation.py --dataset_path /gpfsscratch/rech/fkc/uhx75if/midi_parseq_dataset --settings_path ./new_deforum_settings.txt
