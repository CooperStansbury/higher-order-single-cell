RNA_ROOT = "/nfs/turbo/umms-indikar/shared/projects/poreC/data/4DN_Features/RNAseq/"
RNA_batches = [
    '4DNFI8CSCJWM',
    '4DNFICXJQ3PA',
    '4DNFI3YYNDKI',
    '4DNFIPYGE7JR',
    '4DNFIYTCHMIZ',
    '4DNFIC269AEU',
]

rna_pool_ROOT = "/nfs/turbo/umms-indikar/shared/projects/poreC/data/4DN_Features/rna_bigwig/"
rna_pool_batches = [
    '4DNFI12AUKQS',
    '4DNFIFVPB94O',
    '4DNFIUW8CG2I',
    '4DNFIXOTRTRM',
    '4DNFI4XVSIFH',
    '4DNFIW5IZKYG',
]

ATAC_ROOT = "/nfs/turbo/umms-indikar/shared/projects/poreC/data/4DN_Features/ATACSeq/"
ATAC_batches = [
    '4DNFIPVAKPXA',
    '4DNFIXT1TVT4',
    '4DNFI3ARZKH6',
]
          
CTCF_ROOT = "/nfs/turbo/umms-indikar/shared/projects/poreC/data/4DN_Features/CTCF/"
CTCF_batches = [
    '4DNFIFFXFV82',
]   

H3K27me3_ROOT = "/nfs/turbo/umms-indikar/shared/projects/poreC/data/4DN_Features/H3K27me3/"
H3K27me3_batches = [
    '4DNFI7UN2C36',
]     

H3K27ac_ROOT = "/nfs/turbo/umms-indikar/shared/projects/poreC/data/4DN_Features/H3K27ac/"
H3K27ac_batches = [
    '4DNFIXE23VC7',
]    


rule get_rna:
    input:
        gene_table=OUTPUT + "reference/gene_table.parquet",
        tables = expand(RNA_ROOT + "{id}.tsv", id=RNA_batches),
    output:
        OUTPUT + "rna/expression.parquet"
    conda:
        'bioinf'
    shell:
        """python scripts/get_expression_data.py {input.gene_table} {output} {input.tables} """
        



rule get_rna_pool_data:
    input:
        tables = expand(rna_pool_ROOT + "{id}.bw", id=rna_pool_batches),
    output:
        OUTPUT + "1D_features/RNA_{chr}_{res}.parquet"
    conda:
        'bioinf'
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
        """python scripts/bigwig_to_df.py {output} {wildcards.chr} {wildcards.res} {input.tables} """



rule get_atac_data:
    input:
        tables = expand(ATAC_ROOT + "{id}.bw", id=ATAC_batches),
    output:
        OUTPUT + "1D_features/ATAC_{chr}_{res}.parquet"
    conda:
        'bioinf'
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
        """python scripts/bigwig_to_df.py {output} {wildcards.chr} {wildcards.res} {input.tables} """
        
        
         
rule get_ctcf_data:
    input:
        tables = expand(CTCF_ROOT + "{id}.bw", id=CTCF_batches),
    output:
        OUTPUT + "1D_features/CTCF_{chr}_{res}.parquet"
    conda:
        'bioinf'
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
        """python scripts/bigwig_to_df.py {output} {wildcards.chr} {wildcards.res} {input.tables} """
        
        
  
rule get_H3K27me3_data:
    input:
        tables = expand(H3K27me3_ROOT + "{id}.bw", id=H3K27me3_batches),
    output:
        OUTPUT + "1D_features/H3K27me3_{chr}_{res}.parquet"
    conda:
        'bioinf'
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
        """python scripts/bigwig_to_df.py {output} {wildcards.chr} {wildcards.res} {input.tables} """
        
           
rule get_H3K27ac_data:
    input:
        tables = expand(H3K27ac_ROOT + "{id}.bw", id=H3K27ac_batches),
    output:
        OUTPUT + "1D_features/H3K27ac_{chr}_{res}.parquet"
    conda:
        'bioinf'
    wildcard_constraints:
        chr='|'.join([re.escape(x) for x in set(chrom_names)]),
        res='|'.join([re.escape(x) for x in set(resolutions)]),     
    shell:
        """python scripts/bigwig_to_df.py {output} {wildcards.chr} {wildcards.res} {input.tables} """
        