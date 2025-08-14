rule get_chrom_sizes:
    input:
        config['chrom_sizes']
    output:
        OUTPUT + "reference/chrom_sizes.csv"
    conda:
        'higher_order'
    params:
        chroms=",".join(config['chromosomes'])
    shell:
        """python scripts/get_chromsizes.py {input} {params.chroms} {output}"""
        
        

rule get_scenic:
    input:
        config['scenic_path']
    output:
        OUTPUT + "reference/scenic.parquet"
    conda:
        'bioinf'
    shell:
        """python scripts/get_scenic.py {input} {output}"""
        
        
rule get_annotations:
    input:
        config['gtf_path']
    output:
        OUTPUT + "reference/gene_annotations.gtf.gz"
    shell:
        """cp {input} {output}"""
        

rule get_gene_annotations:
    input:
        gtf=OUTPUT + "reference/gene_annotations.gtf.gz",
        scenic=OUTPUT + "reference/scenic.parquet",
    output:
        OUTPUT + "reference/gene_table.parquet",
    conda:
        'bioinf'
    shell:
        """python scripts/get_gtf.py {input.gtf} {input.scenic} {output}"""

                