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
#SBATCH --nodelist=kp067
lscpu
echo "mmt trial 1"
./mmt | tee -a kingspeak_msort.$SLURM_JOB_ID\.log
echo "mmt trial 2"
./mmt | tee -a kingspeak_msort.$SLURM_JOB_ID\.log
echo "mmt trial 3"
./mmt | tee -a kingspeak_msort.$SLURM_JOB_ID\.log
