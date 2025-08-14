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

# Parse command-line argument
RULE_NAME=""
while getopts ":R:" opt; do
  case $opt in
    R)
      RULE_NAME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Run snakemake with optional rule name
if [ -z "$RULE_NAME" ]; then
  # Default behavior if no -R is provided
  snakemake --profile ${CONFIG} \
  --use-conda \
  --cores ${CORES} \
  --rerun-triggers mtime \
  --rerun-incomplete \
  --latency-wait 90 \
  --keep-going \
  --verbose \
  -s workflow.smk 
else
  # Run a specific rule
  snakemake --profile ${CONFIG} \
  --use-conda \
  --cores ${CORES} \
  --rerun-triggers mtime \
  --rerun-incomplete \
  --latency-wait 90 \
  --keep-going \
  --verbose \
  -s workflow.smk \
  -R $RULE_NAME
fi