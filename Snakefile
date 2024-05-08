import sys
import os
import pandas as pd

source_path = os.path.abspath("source/")
sys.path.append(source_path)
import utils as ut

OUTPUT = "/scratch/indikar_root/indikar1/shared_data/higher_order/"
resolutions = ut.read_csv("config/resolutions.txt")['value'].values
chromosomes = ut.read_csv("config/chromosomes.txt")['value'].values
chrom_names = [f"chr{x}" for x in chromosomes]
sc_hic_ids = ut.read_csv("config/single_cell_hic.txt")['value'].values
sc_porec_ids = ut.read_csv("config/single_cell_porec.txt")['value'].values

print(f"{resolutions}")
print(f"{chrom_names}")
print(f"{sc_hic_ids}")
print(f"{sc_porec_ids}")



rule all:
    input:
        f"{OUTPUT}reference/sc_hic_fends.csv",
        f"{OUTPUT}reference/pop_hic_chromsizes.csv",
        expand(f"{OUTPUT}population_hic/{{chr}}_{{res}}.csv", chr=chrom_names, res=resolutions),
        expand(f"{OUTPUT}population_pore_c/{{chr}}_{{res}}_incidence.csv", chr=chrom_names, res=resolutions),
        expand(f"{OUTPUT}sc_hic/{{schic_id}}_{{chr}}_{{res}}.csv", schic_id=sc_hic_ids, chr=chrom_names, res=resolutions),
        expand(f"{OUTPUT}sc_porec/{{scporec_id}}_{{chr}}_{{res}}.csv", scporec_id=sc_porec_ids, chr=chrom_names, res=resolutions),

rule get_pop_hic:
    input:
        matrix="/nfs/turbo/umms-indikar/shared/projects/poreC/data/f1219_population_hic/4DNFICF9PA9C.mcool",
        res="config/resolutions.txt",
        chroms="config/chromosomes.txt",
    output:
        f"{OUTPUT}population_hic/{{chr}}_{{res}}.csv"
    shell:
        """python scripts/get_population_hic.py {input.matrix} {wildcards.res} {wildcards.chr} {output}"""
        

rule get_chromsizes:
    input:
        "/nfs/turbo/umms-indikar/shared/projects/poreC/data/f1219_population_hic/4DNFICF9PA9C.mcool"
    output:
        f"{OUTPUT}reference/pop_hic_chromsizes.csv"
    shell:
        """python scripts/get_chromsizes.py {input} {output}"""
        

                
PORE_C_ROOT = "/scratch/indikar_root/indikar1/shared_data/population_pore_c/align_table/"
PORE_C_BATCHES = ['batch01', 'batch02', 'batch03', 'batch04']
rule get_population_pore_c:
    input:
        tables = expand(f"{PORE_C_ROOT}{{batch}}.GRCm39.align_table.parquet", batch=PORE_C_BATCHES),
        res="config/resolutions.txt",
        chroms="config/chromosomes.txt",
    output:
        f"{OUTPUT}population_pore_c/{{chr}}_{{res}}_incidence.csv"
    shell:
         """python scripts/get_population_pore_c.py {wildcards.res} {wildcards.chr} {output} {input.tables}""" 
        
 
rule get_schic_fends:
    input:
        fends="/nfs/turbo/umms-indikar/shared/projects/poreC/data/nagano2017/schic2_mm9/seq/redb/GATC.fends",
    output:
        f"{OUTPUT}reference/sc_hic_fends.csv"
    shell:
        "cp {input} {output}"
        
        
rule get_sc_hic:
    input:
        mat="/nfs/turbo/umms-indikar/shared/projects/poreC/data/nagano2017/matrices/all_mats/{schic_id}.csv",
        fend=f"{OUTPUT}reference/sc_hic_fends.csv",
    output:
        f"{OUTPUT}sc_hic/{{schic_id}}_{{chr}}_{{res}}.csv"
    shell:
        """python scripts/get_sc_hic.py {input.mat} {input.fend} {wildcards.res} {wildcards.chr} {output} """
        
        

rule get_sc_porec:
    input:
        table="/scratch/indikar_root/indikar1/shared_data/scpore_c/align_table/{sc_porec_id}.GRCm39.align_table.parquet",
    output:
        f"{OUTPUT}sc_porec/{{sc_porec_id}}_{{chr}}_{{res}}.csv"
    shell:
        """python scripts/get_sc_porec.py {input.table} {wildcards.res} {wildcards.chr} {output} """
        