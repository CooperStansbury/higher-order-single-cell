rule copy_references:
    input:
        ref_df['file_path'].to_list()
    output:
        ref_df['out_path'].to_list(),
    run:
        from shutil import copyfile
        for i, fpath in enumerate(input):
    
            outPath = output[i]
            copyfile(fpath, outPath)


rule make_chromsizes:
    input:
        OUTPUT + 'references/{rid}.fa',
    output:
        OUTPUT + 'references/{rid}.chrom.sizes'
    wildcard_constraints:
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "bioinf"
    shell:
        "faidx {input} -i chromsizes > {output}"


rule digest_reference:
    input:
        ref=OUTPUT + 'references/{rid}.fa',
        sizes=OUTPUT + 'references/{rid}.chrom.sizes',
    output:
        OUTPUT + 'references/{rid}.fragments.bed'
    wildcard_constraints:
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "cooler"
    params:
        enzyme=config['enzyme']
    shell:
        """cooler digest -o {output} {input.sizes} {input.ref} {params.enzyme}"""


rule minimap2_index:
    input:
        refgenome=OUTPUT + 'references/{rid}.fa'
    output:
        OUTPUT + 'references/{rid}.mmi'
    threads:
        config['threads'] // 4
    wildcard_constraints:
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "aligner"
    shell:
        "minimap2 -t {threads} -d {output} {input.refgenome}"


rule copy_gtfs:
    input:
        gtf_df['file_path'].to_list()
    output:
        gtf_df['out_path'].to_list(),
    run:
        from shutil import copyfile
        for i, fpath in enumerate(input):
    
            outPath = output[i]
            copyfile(fpath, outPath)


rule build_gene_table:
    input:
        OUTPUT + "gtf/{rid}.gtf.gz",
    output:
        OUTPUT + "gtf/{rid}.gene_table.parquet",
    wildcard_constraints:
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "bioinf"
    shell:
        """python scripts/build_gene_table.py {input} {output}"""
        

rule build_fragment_db:
    input:
        OUTPUT + 'references/{rid}.fragments.bed'
    output:
        OUTPUT + "references/{rid}.fragment_db.parquet",
    wildcard_constraints:
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "bioinf"
    shell:
        """python scripts/make_fragment_database.py {input} {output}"""
    
