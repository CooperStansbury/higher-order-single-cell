import sys
import os
import re
import pandas as pd

source_path = os.path.abspath("source/")
sys.path.append(source_path)
import utils as ut

# GLOBAL VARIABLES
OUTPUT = "/scratch/indikar_root/indikar1/shared_data/higher_order/"
PORE_C_ROOT = "/scratch/indikar_root/indikar1/shared_data/population_pore_c/align_table/"
PORE_C_BATCHES = ['batch01', 'batch02', 'batch03', 'batch04']
         
resolutions = ut.read_csv("config/resolutions.txt")['value'].astype(str).str.strip().values
chromosomes = ut.read_csv("config/chromosomes.txt")['value'].astype(str).str.strip().values
chrom_names = ["chr" + str(x) for x in chromosomes]
sc_hic_ids = ut.read_csv("config/single_cell_hic.txt")['value'].astype(str).str.strip().values
sc_porec_ids = ut.read_csv("config/single_cell_porec.txt")['value'].astype(str).str.strip().values

print(f"{resolutions}")
print(f"{chrom_names}")
print(f"{sc_hic_ids}")
print(f"{sc_porec_ids}")


##########################################################################################
### RULES
##########################################################################################

rule all:
    input:
        OUTPUT + "reference/sc_hic_fends.csv",
        OUTPUT + "reference/pop_hic_chromsizes.parquet",
        OUTPUT + "reference/scenic.parquet",
        OUTPUT + "reference/gene_table.parquet",
        OUTPUT + "rna/expression.parquet",
        expand(OUTPUT + "population_pore_c/{chr}_genes_incidence.parquet", chr=chrom_names),
        expand(OUTPUT + "1D_features/ATAC_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        expand(OUTPUT + "1D_features/CTCF_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        expand(OUTPUT + "1D_features/H3K27me3_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        expand(OUTPUT + "1D_features/H3K27ac_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        expand(OUTPUT + "1D_features/RNA_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        expand(OUTPUT + "population_hic/{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        expand(OUTPUT + "population_pore_c/{chr}_{res}_incidence.parquet", chr=chrom_names, res=resolutions),
        expand(OUTPUT + "sc_hic/{schic_id}_{chr}_{res}.parquet", schic_id=sc_hic_ids, chr=chrom_names, res=resolutions),
        expand(OUTPUT + "sc_porec/{scporec_id}_{chr}_{res}.parquet", scporec_id=sc_porec_ids, chr=chrom_names, res=resolutions),



rule get_pop_hic:
    input:
        matrix="/nfs/turbo/umms-indikar/shared/projects/poreC/data/f1219_population_hic/4DNFICF9PA9C.mcool",
        res="config/resolutions.txt",
        chroms="config/chromosomes.txt",
    output:
        OUTPUT + "population_hic/{chr}_{res}.parquet"
    conda:
        "cooler"
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
        """python scripts/get_population_hic.py {input.matrix} {wildcards.res} {wildcards.chr} {output}"""
        

rule get_chromsizes:
    input:
        "/nfs/turbo/umms-indikar/shared/projects/poreC/data/f1219_population_hic/4DNFICF9PA9C.mcool"
    output:
        OUTPUT + "reference/pop_hic_chromsizes.parquet"
    conda:
        "cooler"
    shell:
        """python scripts/get_chromsizes.py {input} {output}"""
        

rule get_scenic:
    input:
        "/nfs/turbo/umms-indikar/shared/projects/DGC/data/processed_data/MOUSE_500bp_up_100bp_down_B.csv"
    output:
        OUTPUT + "reference/scenic.parquet"
    conda:
        'bioinf'
    shell:
        """python scripts/get_scenic.py {input} {output}"""
        

rule get_gene_annotations:
    input:
        gtf="/scratch/indikar_root/indikar1/shared_data/scpore_c/gtf/GRCm39.gtf.gz",
        scenic=OUTPUT + "reference/scenic.parquet",
    output:
        OUTPUT + "reference/gene_table.parquet"
    conda:
        'bioinf'
    shell:
        """python scripts/get_gtf.py {input.gtf} {input.scenic} {output}"""

                
rule get_population_pore_c:
    input:
        tables = expand(PORE_C_ROOT + "{batch}.GRCm39.align_table.parquet", batch=PORE_C_BATCHES),
        res="config/resolutions.txt",
        chroms="config/chromosomes.txt",
    output:
        OUTPUT + "population_pore_c/{chr}_{res}_incidence.parquet"
    conda:
        "higher_order"
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
         """python scripts/get_population_pore_c.py {wildcards.res} {wildcards.chr} {output} {input.tables}""" 
         
         
rule population_pore_c_genes:
    input:
        gene_table=OUTPUT + "reference/gene_table.parquet",
        tables = expand(PORE_C_ROOT + "{batch}.GRCm39.align_table.parquet", batch=PORE_C_BATCHES),
        chroms="config/chromosomes.txt",
    output:
        OUTPUT + "population_pore_c/{chr}_genes_incidence.parquet"
    conda:
        "higher_order"
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
         """python scripts/population_pore_c_gene_edges.py {input.gene_table} {wildcards.chr} {output} {input.tables}""" 
         

 
rule get_schic_fends:
    input:
        fends="/nfs/turbo/umms-indikar/shared/projects/poreC/data/nagano2017/schic2_mm9/seq/redb/GATC.fends",
    output:
        OUTPUT + "reference/sc_hic_fends.csv"
    shell:
        "cp {input} {output}"


rule get_sc_hic:
    input:
        mat="/nfs/turbo/umms-indikar/shared/projects/poreC/data/nagano2017/matrices/all_mats/{schic_id}.csv",
        fend=OUTPUT + "reference/sc_hic_fends.csv",
    output:
        OUTPUT + "sc_hic/{schic_id}_{chr}_{res}.parquet"
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    conda:
        "higher_order"
    shell:
        """python scripts/get_sc_hic.py {input.mat} {input.fend} {wildcards.res} {wildcards.chr} {output} """
        
        

rule get_sc_porec:
    input:
        table="/scratch/indikar_root/indikar1/shared_data/single_cell/align_table/{sc_porec_id}.GRCm39.align_table.parquet",
    output:
        OUTPUT + "sc_porec/{sc_porec_id}_{chr}_{res}.parquet"
    conda:
        "higher_order"
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
        """python scripts/get_sc_porec.py {input.table} {wildcards.res} {wildcards.chr} {output} """
        