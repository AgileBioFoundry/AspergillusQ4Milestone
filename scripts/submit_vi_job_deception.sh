#!/bin/bash
#SBATCH --account=agilebiofoundry
#SBATCH --partition=slurm
#SBATCH --time=4-00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=run_bmca
#SBATCH --output=/qfs/projects/agilebiofoundry/aspergillus_niger_round1_pytensor/AspergillusQ4Milestone/data/runs/round1/%j.%x.out  # %j will be replaced with the job ID
#SBATCH --error=/qfs/projects/agilebiofoundry/aspergillus_niger_round1_pytensor/AspergillusQ4Milestone/data/runs/round1/%j.%x.err  # %j will be replaced with the job ID
#SBATCH --mail-user=august.george@pnnl.gov
#SBATCH --mail-type=ALL



#module load python/miniconda3.9
conda init

#choose theano or pytensor env
#conda activate /qfs/projects/nwbrave/lattice_microbes/v2.4/env/emll_env
conda activate /qfs/projects/nwbrave/lattice_microbes/v2.4/env/emll_pytensor_env

# move to the source directory
cd /qfs/projects/agilebiofoundry/aspergillus_niger_round1_pytensor/AspergillusQ4Milestone/src

echo "Activated conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# round 1
# choose to run theano or pytensor model
#srun python run_inference_theano_on_round1.py
srun python run_inference_pytensor_on_round1.py


# round 2 (reduced)
# choose to run theano or pytensor model
#srun python run_inference_theano_on_round2_reduced.py
#srun python run_inference_pytensor_on_round2_reduced.py