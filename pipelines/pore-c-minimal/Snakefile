import pandas as pd
import yaml
from pathlib import Path
import re
import os
import sys
from utils import utils
from snakemake.utils import Paramspace
from tabulate import tabulate

BASE_DIR = Path(workflow.basedir)
configfile: str(BASE_DIR) + "/config/config.yaml"
OUTPUT = config['output_path']

# load the fastq files 
fastq_paths = os.path.abspath(config['fastq_paths'])
fastq_df = utils.load_fastq_df(fastq_paths, OUTPUT)
cell_ids = fastq_df['cell_id'].to_list() # wildcard constraints

print(f"\n======== INPUT FILES ========")
print(tabulate(fastq_df[['cell_id', 'basename']], 
      headers='keys',
      showindex=False,
      tablefmt='psql'))

# build the references
ref_paths = os.path.abspath(config['ref_paths'])
ref_df = utils.load_ref_df(ref_paths, OUTPUT)
ref_ids = ref_df['ref_id'].to_list() # wildcard constraints

print(f"\n======== REFERENCES ========")
print(tabulate(ref_df[['ref_id', 'basename']], 
      headers='keys',
      showindex=False,
      tablefmt='psql'))


# build the annotations
gtf_paths = os.path.abspath(config['gtf_paths'])
gtf_df = utils.load_gtf_df(gtf_paths, OUTPUT)
gtf_ids = gtf_df['gtf_id'].to_list() # wildcard constraints

print(f"\n======== GTF ========")
print(tabulate(gtf_df[['gtf_id', 'basename']], 
      headers='keys',
      showindex=False,
      tablefmt='psql'))


################ RULE FILES ################
include: "rules/references.smk"
include: "rules/core.smk"
include: "rules/library.smk"
include: "rules/reports.smk"
include: "rules/ensemble.smk"
include: "rules/single_cell.smk"
  
################ ALL RULES ################
rule all:
    input:
        expand(OUTPUT + "references/{rid}.mmi", rid=ref_ids),
        expand(OUTPUT + "references/{rid}.fragment_db.parquet", rid=ref_ids),
        expand(OUTPUT + "fastq/{cid}.raw.fastq", cid=cell_ids),
        expand(OUTPUT + "references/{rid}.chrom.sizes", rid=ref_ids),
        expand(OUTPUT + "references/{rid}.fragments.bed", rid=ref_ids),
        expand(OUTPUT + "alignments/{cid}.{rid}.bam", cid=cell_ids, rid=ref_ids),
        expand(OUTPUT + "digest/{cid}.{rid}.bam", cid=cell_ids, rid=ref_ids),
        expand(OUTPUT + "align_table/{cid}.{rid}.align_table.parquet", cid=cell_ids, rid=ref_ids),
        expand(OUTPUT + "annotate/{cid}.{rid}.ns.bam", cid=cell_ids, rid=ref_ids),
        expand(OUTPUT + "pairs/{cid}.{rid}.expanded.pairs", cid=cell_ids, rid=ref_ids),
        expand(OUTPUT + "pairs/{cid}.{rid}.direct.pairs", cid=cell_ids, rid=ref_ids),


rule library:
    input:
        OUTPUT + "library/merged.bam",
        OUTPUT + "library/merged.duplicates.bam",
        OUTPUT + "reports/samtools/library_coverage.txt",
        OUTPUT + "reports/samtools/flagstat.txt",
        expand(OUTPUT + "reports/read_lengths/{cid}.read_lengths.parquet", cid=cell_ids),        
 
rule reports:
    input:
        OUTPUT + "reports/seqkit/fastq.report.txt",
        OUTPUT + "reports/mapping_percentage.csv",
        OUTPUT + "reports/monomer_mapping/mapping_summary.csv",
        expand(OUTPUT + "reports/samtools/flagstat/{cid}.{rid}.digest.txt", cid=cell_ids, rid=ref_ids),
        expand(OUTPUT + "reports/samtools/coverage/{cid}.{rid}.digest.txt", cid=cell_ids, rid=ref_ids),
        OUTPUT + "reports/annotate/cardinality.csv",
        OUTPUT + "reports/pairtools/merged.expanded.stats",
        OUTPUT + "reports/pairtools/merged.direct.stats",
        OUTPUT + "reports/pairs/direct.basic.csv",
        OUTPUT + "reports/pairs/expanded.basic.csv",
        
       
rule ensemble:
    input:
        OUTPUT + "ensemble/expanded.dedup.pairs",
        OUTPUT + "ensemble/direct.dedup.pairs",
        expand(OUTPUT + "cooler/expanded.{rid}.mcool", rid=ref_ids),
        expand(OUTPUT + "cooler/direct.{rid}.mcool", rid=ref_ids),


rule single_cell:
    input:
        OUTPUT + "reports/single_cell/direct.filter_report.csv",
        OUTPUT + "reports/single_cell/expanded.filter_report.csv",
        OUTPUT + "reports/single_cell/pairtools/direct.stats",
        OUTPUT + "reports/single_cell/pairtools/expanded.stats",
        OUTPUT + "reports/single_cell/pairs/direct.basic.csv",
        OUTPUT + "reports/single_cell/pairs/expanded.basic.csv",
        OUTPUT + "reports/single_cell/duplication_report.csv",
        expand(OUTPUT + "single_cell/direct/{cid}.{rid}.filtered.pairs", cid=cell_ids, rid=ref_ids),
        expand(OUTPUT + "single_cell/expanded/{cid}.{rid}.filtered.pairs", cid=cell_ids, rid=ref_ids),
        expand(OUTPUT + "single_cell/duplication_reports/{cid}.{rid}.dupes.parquet", cid=cell_ids, rid=ref_ids),
        
rule gtf:
    input:
        expand(OUTPUT + "gtf/{gid}.gtf.gz", gid=gtf_ids),
        expand(OUTPUT + "gtf/{gid}.gene_table.parquet", gid=gtf_ids),
