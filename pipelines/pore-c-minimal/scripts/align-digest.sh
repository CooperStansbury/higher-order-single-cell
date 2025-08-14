#!/bin/bash

# a script to digest and align a set of sequences
# based on https://github.com/epi2me-labs/pore-c-py

FASTQ=""
BAM=""
REF=""
ENZYME='NlaIII'  # Default value
THREADS=4
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--fastq)
      shift
      FASTQ="$1"
      ;;
     -b|--bam)
      shift
      BAM="$1"
      ;;
    -r|--reference)
      shift
      REF="$1"
      ;;
    -e|--enzyme)
      shift
      ENZYME="$1"
      ;; 
    -t|--threads)
      shift
      THREADS="$1"
      ;;
    -o|--output)
      shift
      OUTPUT="$1"
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
  shift
done

echo "Input fastq:"  "${FASTQ}"
echo "Input BAM:"  "${BAM}"
echo "Input reference:"  "${REF}"
echo "Restriction enzyme:" "${ENZYME}"
echo "Threads:" "${THREADS}"
echo "Output:" "${OUTPUT}"

samtools view -S -bS "${FASTQ}" \
| pore-c-py digest "${ENZYME}" --remove_tags 'Mm' --header "${BAM}" --threads "${THREADS}" \
| samtools fastq -T '*' \
| minimap2 -ay -t "${THREADS}" -x map-ont "${REF}" - > "${OUTPUT}"
