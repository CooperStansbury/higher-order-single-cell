#!/bin/bash

#SBATCH --account=indikar99
#SBATCH --partition=standard
#SBATCH --mail-user=cstansbu@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --time=36:00:00

CONFIG='config/cluster'
CORES=36

### export the environment 
conda env export > environment.yml

## build the workflow from the most current snakefile
cp Snakefile workflow.smk

# run it
snakemake --profile ${CONFIG} --use-conda --cores ${CORES} --rerun-triggers mtime --rerun-incomplete --latency-wait 90 --verbose -s workflow.smk 