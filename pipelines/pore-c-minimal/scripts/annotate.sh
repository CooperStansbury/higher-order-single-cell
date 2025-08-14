#!/bin/bash

# Annotate aligned monomers using pore-c-py
# (based on https://github.com/epi2me-labs/pore-c-py)
# with support for user-defined flags

# --- Functions ---

usage() {
    echo "Usage: $0 -b <input.bam> -o <output.ns.bam> [-t <threads>] [--direct-only] [-- <pore-c-py flags>]"
    echo "Example: $0 -b input.bam -o output.ns.bam -- --min-seq-len 100 --max-seq-len 1500"  # Example usage with flags
    exit 1
}

# --- Main Script ---

# Parse command-line options with error handling
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b | --bam)
            shift
            BAM="$1" || usage
            ;;
        -o | --output)
            shift
            OUTPUT="$1" || usage
            ;;
        --)  # End of script's options, rest are for pore-c-py
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
    shift
done

# Input validation
if [[ -z "$BAM" || -z "$OUTPUT" ]]; then
    usage
fi

# Build pore-c-py command with user-provided flags
PORECPY_CMD="pore-c-py annotate $BAM $OUTPUT --force --paired_end --monomers --summary --stdout"

# Add any remaining arguments as pore-c-py flags
if [[ $# -gt 0 ]]; then
    PORECPY_CMD+=" $*"  # Append the rest of the arguments
fi

# Execute pore-c-py with the constructed command
eval "$PORECPY_CMD" > "${OUTPUT}".ns.bam

echo "Annotation complete. Output saved to ${OUTPUT}.ns.bam"  # User-friendly confirmation
