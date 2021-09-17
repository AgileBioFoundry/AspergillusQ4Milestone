#!/bin/bash
#SBATCH --account=bpms
#SBATCH --qos=high
#SBATCH --time=2-00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=run_advi
#SBATCH --output=/scratch/pstjohn/%j.%x.out  # %j will be replaced with the job ID

module unload
unset LD_PRELOAD

module load singularity-container/3.2.1

SIMG=/projects/bpms/pstjohn/containers/emll.simg
SINGULARTY_CMD="singularity exec -B /scratch,/projects --nv $SIMG"

srun $SINGULARTY_CMD python run_advi_large.py --lr=0.01
