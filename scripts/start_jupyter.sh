#!/bin/bash

module unload
unset LD_PRELOAD

module load singularity-container

singularity exec -B /scratch,/projects /projects/bpms/pstjohn/containers/emll.simg start.sh jupyter lab --no-browser --ip=0.0.0.0

