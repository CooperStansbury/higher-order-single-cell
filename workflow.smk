import sys
import os
import re
import pandas as pd
import glob
import tabulate

source_path = os.path.abspath("source/")
sys.path.append(source_path)

BASE_DIR = Path(workflow.basedir)
configfile: str(BASE_DIR) + "/config/config.yaml"

# Print the config
print("\n----------------- CONFIG VALUES -----------------")
for key, value in config.items():
    print(f"{key}: {value}")  # Adjust spacing as needed

# GLOBAL VARIABLES
OUTPUT = config['outpath']
resolutions = [int(x) for x in config['resolutions']]

# load in feature paths
feature_paths = os.path.abspath(config['feature_paths'])
feature_paths = pd.read_csv(feature_paths, comment="#")
feature_ids = feature_paths['file_id'].to_list()

# get new path names
feature_output_paths = [OUTPUT + "features/" + x + ".bw" for x in feature_ids]
    
print("\n----- FEATURE VALUES -----")
print(
    tabulate.tabulate(
        feature_paths, 
        headers='keys', 
        tablefmt='psql',
        showindex=False,
    )
)

##################################
### SUPPLEMENTAL RULE FILES
##################################
include: "rules/references.smk"

##################################
### RULES
##################################

rule all:
    input:
        OUTPUT + "reference/chrom_sizes.csv",
        OUTPUT + "reference/scenic.parquet",
        OUTPUT + "reference/gene_table.parquet",
        OUTPUT + "pore_c/population_mESC.read_level.parquet",
        OUTPUT + "pore_c/singlecell_mESC.read_level.parquet",
        expand(OUTPUT + "features/{fid}.bw", fid=feature_ids),
        expand(OUTPUT + "anndata/{level}_{resolution}_raw.h5ad", level=['population_mESC', 'singlecell_mESC'], resolution=resolutions),
        expand(OUTPUT + "anndata/{level}_{resolution}_features.h5ad", level=['population_mESC', 'singlecell_mESC'], resolution=resolutions),
        # OUTPUT + "reference/sc_hic_fends.csv",
        # expand(OUTPUT + "1D_features/ATAC_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        # expand(OUTPUT + "1D_features/CTCF_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        # expand(OUTPUT + "1D_features/H3K27me3_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        # expand(OUTPUT + "1D_features/H3K27ac_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        # expand(OUTPUT + "1D_features/RNA_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        # expand(OUTPUT + "population_hic/{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
        # expand(OUTPUT + "sc_hic/{schic_id}_{chr}_{res}.parquet", schic_id=sc_hic_ids, chr=chrom_names, res=resolutions),



rule gather_linear_features:
    input:
        feature_paths['file_path'].to_list()
    output:
        feature_output_paths
    run:
        from shutil import copyfile
        for i, refPath in enumerate(input):

            outPath = output[i]
            copyfile(refPath, outPath)


rule get_population_pore_c:
    input:
        chrom_path=OUTPUT + "reference/chrom_sizes.csv",
        file_list=glob.glob(config['pop_pore_c_path'] + "*"),
    output:
        OUTPUT + "pore_c/population_mESC.read_level.parquet",
    conda:
        'higher_order'
    shell:
        """python scripts/get_pore_c.py {input.chrom_path} \
        {output} {input.file_list}"""
        

rule get_sc_pore_c:
    input:
        chrom_path=OUTPUT + "reference/chrom_sizes.csv",
        file_list=glob.glob(config['sc_pore_c_path'] + "*"),
    output:
        OUTPUT + "pore_c/singlecell_mESC.read_level.parquet",
    conda:
        'higher_order'
    shell:
        """python scripts/get_pore_c.py {input.chrom_path} \
        {output} {input.file_list}"""
    

rule make_pore_c_anndata:
    input:
        porec=OUTPUT + "pore_c/{level}.read_level.parquet",
        chroms=OUTPUT + "reference/chrom_sizes.csv",
        genes=OUTPUT + "reference/gene_table.parquet",
    output:
        anndata=OUTPUT + "anndata/{level}_{resolution}_raw.h5ad",
        log=OUTPUT + "reports/anndata_logs/{level}_{resolution}_raw.txt",
    conda:
        "scanpy"
    shell:
        """
        python scripts/make_anndata.py {input.porec} \
                                       {wildcards.resolution} \
                                       {input.chroms} \
                                       {input.genes} \
                                       {output.anndata} > {output.log}
        """



rule add_features:
    input:
        anndata=OUTPUT + "anndata/{level}_{resolution}_raw.h5ad",
        features=expand(feature_output_paths),
    output:
        anndata=OUTPUT + "anndata/{level}_{resolution}_features.h5ad",
        log=OUTPUT + "reports/anndata_logs/{level}_{resolution}_features.txt",
    conda:
        "scanpy"
    shell:
        """python scripts/add_features.py {output.anndata} {input.anndata} {input.features} > {output.log} """
        





# rule get_pop_hic:
#     input:
#         matrix="/nfs/turbo/umms-indikar/shared/projects/poreC/data/f1219_population_hic/4DNFICF9PA9C.mcool",
#         res="config/resolutions.txt",
#         chroms="config/chromosomes.txt",
#     output:
#         OUTPUT + "population_hic/{chr}_{res}.parquet"
#     conda:
#         "cooler"
#     wildcard_constraints:
#         chr='|'.join([re.escape(x) for x in set(chrom_names)]),
#         res='|'.join([re.escape(x) for x in set(resolutions)]),     
#     shell:
#         """python scripts/get_population_hic.py {input.matrix} {wildcards.res} {wildcards.chr} {output}"""
#                
# rule population_pore_c_genes:
#     input:
#         gene_table=OUTPUT + "reference/gene_table.parquet",
#         tables = expand(PORE_C_ROOT + "{batch}.GRCm39.align_table.parquet", batch=PORE_C_BATCHES),
#         chroms="config/chromosomes.txt",
#     output:
#         OUTPUT + "population_pore_c/{chr}_genes_incidence.parquet"
#     conda:
#         "higher_order"
#     wildcard_constraints:
#         chr='|'.join([re.escape(x) for x in set(chrom_names)]),
#         res='|'.join([re.escape(x) for x in set(resolutions)]),     
#     shell:
#          """python scripts/population_pore_c_gene_edges.py {input.gene_table} {wildcards.chr} {output} {input.tables}""" 
#          
# 
#  
# rule get_schic_fends:
#     input:
#         fends="/nfs/turbo/umms-indikar/shared/projects/poreC/data/nagano2017/schic2_mm9/seq/redb/GATC.fends",
#     output:
#         OUTPUT + "reference/sc_hic_fends.csv"
#     shell:
#         "cp {input} {output}"
# 
# 
# rule get_sc_hic:
#     input:
#         mat="/nfs/turbo/umms-indikar/shared/projects/poreC/data/nagano2017/matrices/all_mats/{schic_id}.csv",
#         fend=OUTPUT + "reference/sc_hic_fends.csv",
#     output:
#         OUTPUT + "sc_hic/{schic_id}_{chr}_{res}.parquet"
#     wildcard_constraints:
#         chr='|'.join([re.escape(x) for x in set(chrom_names)]),
#         res='|'.join([re.escape(x) for x in set(resolutions)]),     
#     conda:
#         "higher_order"
#     shell:
#         """python scripts/get_sc_hic.py {input.mat} {input.fend} {wildcards.res} {wildcards.chr} {output} """