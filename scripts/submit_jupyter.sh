#!/bin/bash
#SBATCH --account=bpms
#SBATCH --time=2-00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=emll_jupyter
#SBATCH --output=/scratch/pstjohn/%j.%x.out  # %j will be replaced with the job ID

module unload
unset LD_PRELOAD

module load singularity-container

SIMG=/projects/bpms/pstjohn/containers/emll.simg
SINGULARTY_CMD="singularity exec -B /scratch,/projects --nv $SIMG"

date      > ~/jupyter.log
hostname >> ~/jupyter.log
srun $SINGULARTY_CMD start.sh jupyter lab --no-browser --ip=0.0.0.0 >> ~/jupyter.log 2>&1
