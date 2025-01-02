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

# hard-coded helper list
levels = ['population_mESC', 'singlecell_mESC']

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
        expand(OUTPUT + "anndata/{level}_{resolution}_raw.h5ad", level=levels, resolution=resolutions),
        expand(OUTPUT + "anndata/{level}_{resolution}_features.h5ad", level=levels, resolution=resolutions),
        expand(OUTPUT + "reports/anndata/{level}_{resolution}_summary.txt", level=levels, resolution=resolutions),
        expand(OUTPUT + "by_chromosome/{level}_{resolution}_chr{chrom}.h5ad", level=levels, resolution=resolutions, chrom=config['chromosomes']), 
        expand(OUTPUT + "core_scores/population_mESC_{resolution}_chr{chrom}.csv", resolution=resolutions, chrom=config['chromosomes']),
        expand(OUTPUT + "lightweight/{level}_{resolution}_anndata.h5ad", level=levels, resolution=resolutions),


# rule archive:
#     input:
#         # OUTPUT + "reference/sc_hic_fends.csv",
#         # expand(OUTPUT + "1D_features/ATAC_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
#         # expand(OUTPUT + "1D_features/CTCF_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
#         # expand(OUTPUT + "1D_features/H3K27me3_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
#         # expand(OUTPUT + "1D_features/H3K27ac_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
#         # expand(OUTPUT + "1D_features/RNA_{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
#         # expand(OUTPUT + "population_hic/{chr}_{res}.parquet", chr=chrom_names, res=resolutions),
#         # expand(OUTPUT + "sc_hic/{schic_id}_{chr}_{res}.parquet", schic_id=sc_hic_ids, chr=chrom_names, res=resolutions),
# 

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
        

rule simple_anndata_report:
    input:
        OUTPUT + "anndata/{level}_{resolution}_features.h5ad",
    output:
        OUTPUT + "reports/anndata/{level}_{resolution}_summary.txt",
    conda:
        "scanpy"
    shell:
        """python  scripts/report_anndata.py {input} > {output}"""



rule partition_by_chromosome:
    input:
        anndata=OUTPUT + "anndata/{level}_{resolution}_features.h5ad",
    output:
        OUTPUT + "by_chromosome/{level}_{resolution}_chr{chrom}.h5ad",
    conda:
        "scanpy"
    params:
        chrom=config['chromosomes']
    shell:
        """
        python scripts/partition_by_chromosome.py {input.anndata} {wildcards.chrom} {output}
        """


rule compute_core_scores_by_chromosome:
    input:
        anndata=OUTPUT + "anndata/{level}_{resolution}_features.h5ad",
    output:
        OUTPUT + "core_scores/{level}_{resolution}_chr{chrom}.csv",
    conda:
        "scanpy"
    params:
        chrom=config['chromosomes']
    shell:
        """python scripts/compute_chrom_core_scores.py {input.anndata} {wildcards.chrom} {output}"""
        


rule make_lightweight:
    input:
        OUTPUT + "anndata/{level}_{resolution}_features.h5ad",
    output:
        genemap=OUTPUT + "lightweight/{level}_{resolution}_gdf.parquet",
        gdf=OUTPUT + "lightweight/{level}_{resolution}_genemap.parquet",
        anndata=OUTPUT + "lightweight/{level}_{resolution}_anndata.h5ad",
    conda:
        """scanpy"""
    shell:
        """python scripts/make_lightweight.py {input} {output.genemap} {output.gdf} {output.anndata}"""

