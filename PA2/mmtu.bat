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
#SBATCH --nodelist=kp077
lscpu
echo "mmtu trial 1"
./mmtu | tee -a kingspeak_mmt.$SLURM_JOB_ID\.log
echo "mmtu trial 2"
./mmtu | tee -a kingspeak_mmt.$SLURM_JOB_ID\.log
echo "mmtu trial 3"
./mmtu | tee -a kingspeak_mmt.$SLURM_JOB_ID\.log
