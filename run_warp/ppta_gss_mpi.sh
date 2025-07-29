#!/bin/bash

#SBATCH --job-name=equad_only_band_8T
#SBATCH --nodes=1               
#SBATCH --ntasks=8   
#SBATCH --cpus-per-task=1          
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --tmp=1G
#SBATCH --output=./log/equad_only_8T_20s_25.log
ml purge

ml -q load conda
conda activate ppta
cd  /fred/oz103/ezahraoui/PPTA/ppta_dr2_noise_analysis


mpirun -np 8  python /fred/oz103/ezahraoui/PPTA/run_wrap/ppta_gss_wrap.py --prfile /fred/oz103/ezahraoui/PPTA/run_wrap/sampling_params/ppta_pol_7psr_byband_noeccoronly.dat
#mpirun echo "sup" 
####sacct --user ezahraou --format="JobName%-20,avevmsize,State,ExitCode"