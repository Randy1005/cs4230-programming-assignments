#!/bin/bash -x
#SBATCH -M kingspeak 
#SBATCH --account=owner-guest 
#SBATCH --partition=kingspeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -C c16
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH -t 0:05:00
lscpu
echo "Histogram trial 1"
./hist | tee -a kingspeak_hist.$SLURM_JOB_ID\.log
echo "Histogram trial 2"
./hist | tee -a kingspeak_hist.$SLURM_JOB_ID\.log
echo "Histogram trial 3"
./hist | tee -a kingspeak_hist.$SLURM_JOB_ID\.log
