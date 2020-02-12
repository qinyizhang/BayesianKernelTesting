#!/bin/bash
#note anything after #SBATCH is a command
#SBATCH --mail-user=qinyi.zhang@stats.ox.ac.uk
#Email you if job starts, completed or failed
#SBATCH --mail-type=ALL
#SBATCH --job-name=BF_2D
#SBATCH --partition=medium
#Choose your partition depending on your requirements
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#Cpus required for each job
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=500
#Memory per cpu in megabytes
#SBATCH --array=0-2099%200
#SBATCH --output=/data/localhost/not-backed-up/qzhang/jobname_%A_%a.txt

python 2DGaussian_part1.py

# print environment variables: 
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Make a directory for output (.txt) and results
mkdir -p /data/ziz/not-backed-up/qzhang/outputs/jobname_${SLURM_ARRAY_JOB_ID}
mkdir -p /data/ziz/not-backed-up/qzhang/results/jobname_${SLURM_ARRAY_JOB_ID}

# Move experiment outputs & results to the directories made above
mv /data/localhost/not-backed-up/qzhang/jobname_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt /data/ziz/not-backed-up/qzhang/outputs/jobname_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}.txt
mv /data/localhost/not-backed-up/qzhang/jobname_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /data/ziz/not-backed-up/qzhang/results/jobname_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}.out
