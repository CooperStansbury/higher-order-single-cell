#!/bin/bash

# this is a script that merges FASTQ files
# from separate batches into single batches
# ASSUMPTIONS:
# 	(1) that fastq filenames contain "barcodeXX" ids for each cell
# 	(2) that all fastq files with the same barcode should be merged
# 	(3) that `merger.sh` was executed from inside the directory containing files to be merged
# 	(4) barcodes are unique
# 	(5) order does not matter


flag=${1}
echo "Adding" ${flag} "to all files"

barcodeArray=()

for file in *.fastq.gz; do
    n=$(echo $file | grep -oP '(?<=barcode)[0-9]+')
    barcode="barcode${n}"
    barcodeArray+=($barcode)
done

declare -A uniqBarcodes
for i in "${barcodeArray[@]}"; do uniqBarcodes["$i"]=1; done

echo 'Number of FASTQ files:' ${#barcodeArray[@]}

for code in "${!uniqBarcodes[@]}"; do
    echo Processing "$code"    
    cat *"${code}"*.fastq.gz >> "${flag}""${code}".fastq.gz

done

# echo "${!uniqBarcodes[@]}"

