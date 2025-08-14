#!/bin/bash

# Get the input directory from the user or use the current directory
input_dir="${1:-.}"

# Create the output directory if it doesn't exist
output_dir="merged_fastq"
mkdir -p "$output_dir"

# Function to merge FASTQ files within a directory
merge_fastq() {
    local dir_name="$1"
    local output_file="$2"
    
    # Check if files are gzipped
    local is_gzipped=$(file "$dir_name"/* | grep -q gzip && echo true || echo false)

    if [[ "$is_gzipped" == true ]]; then
        # Merge gzipped files using zcat (assumes .gz extension)
        zcat "$dir_name"/* > "$output_file"
    else
        # Merge uncompressed files
        cat "$dir_name"/* > "$output_file"
    fi
}

# Iterate over subdirectories in the input directory
for dir in "$input_dir"/*/; do
    dir_name=$(basename "$dir")
    output_file="$output_dir/$dir_name.fastq"  

    # Merge the files only if the directory is not empty
    if [[ "$(ls -A "$dir")" ]]; then
        merge_fastq "$dir" "$output_file"

        # Check if the output file was gzipped and unzip if so
        if [[ "$is_gzipped" == true ]]; then
            gzip -d "$output_file"
        fi
    fi
done

echo "Merged FASTQ files are located in: $output_dir"