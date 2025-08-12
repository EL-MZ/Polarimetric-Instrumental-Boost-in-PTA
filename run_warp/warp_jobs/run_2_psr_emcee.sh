#!/bin/bash
#SBATCH --job-name=J1600-3053_psr_emcee_backend
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --tmp=1G
#SBATCH --output=./log/J1600-3053_psr_emcee_backend.log

ml purge

ml -q load conda

conda activate ppta

cd  /fred/oz103/ezahraoui/PPTA/ppta_dr2_noise_analysis

# srun echo $TEMPO2
# srun echo $TEMPO2_CLOCK_DIR
python /fred/oz103/ezahraoui/PPTA/run_warp/run_sampler_emcee.py --prfile /fred/oz103/ezahraoui/PPTA/run_warp/sampling_params/2_psr_test_joint_backend.dat

##sacct --user ezahraou --format="JobName%-20,JobID,avevmsize,State,ExitCode"