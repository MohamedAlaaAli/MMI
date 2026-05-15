#!/bin/bash
#SBATCH --account=def-ebrahimi-ab_gpu
#SBATCH --job-name=MMI_train
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00

#SBATCH --output=/home/mohamed0/scratch/projects/multimodal_petct/logs/%x-%j.out
#SBATCH --error=/home/mohamed0/scratch/projects/multimodal_petct/logs/%x-%j.err

echo "============================="
echo "JOB STARTED"
echo "============================="

echo "Node: $(hostname)"
echo "Date: $(date)"

echo "============================="
echo "Cleaning environment"
echo "============================="

module --force purge
module load StdEnv/2023
module load cuda/12.2
module load arrow/24.0.0

echo "============================="
echo "Activating environment"
echo "============================="

source /home/mohamed0/projects/def-ebrahimi-ab/mohamed/MULTIMODALPETCT/MMI/mmi_env/bin/activate

echo "Python:"
which python
python --version

echo "============================="
echo "GPU INFO"
echo "============================="

nvidia-smi

echo "============================="
echo "Moving to project"
echo "============================="

cd /home/mohamed0/projects/def-ebrahimi-ab/mohamed/MULTIMODALPETCT/MMI || exit 1

echo "============================="
echo "Starting training"
echo "============================="

python main.py

EXIT_CODE=$?

echo "============================="
echo "JOB FINISHED"
echo "============================="
echo "Exit code: $EXIT_CODE"
date