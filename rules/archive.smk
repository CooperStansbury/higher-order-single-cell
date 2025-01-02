


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