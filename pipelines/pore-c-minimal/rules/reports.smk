rule get_monomer_mapping_summary:
    input:
        expand(OUTPUT + "align_table/{cid}.{rid}.align_table.parquet", cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + "reports/monomer_mapping/mapping_summary.csv",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "bioinf"
    shell:
        """python scripts/compile_monomer_mapping_summary.py {output} {input}"""
        
        
rule samtools_flagstat_by_sample:
    input:
        OUTPUT + "digest/{cid}.{rid}.bam",
    output:
        OUTPUT + "reports/samtools/flagstat/{cid}.{rid}.digest.txt",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='expanded|direct',
    conda:
        "pore_c"
    shell:
        """samtools flagstat {input} > {output}"""
        
        
rule samtools_coverage_by_sample:
    input:
        OUTPUT + "digest/{cid}.{rid}.bam",
    output:
        OUTPUT + "reports/samtools/coverage/{cid}.{rid}.digest.txt",
    conda:
        "bioinf"
    params:
        temp=OUTPUT,
    shell:
        """samtools sort {input} -T {params.temp} | samtools coverage - > {output}""" 
        
        
rule get_mapping_percentage:
    input:
        OUTPUT + "annotate/{cid}.{rid}.ns.bam",
    output:
        OUTPUT + "reports/mapping_percentage/{cid}.{rid}.txt",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "pore_c"
    shell:
        """samtools flagstat {input} | head -n 7 | tail -n 1 > {output}"""


rule compile_mapping_percentages:
    input:
        expand(OUTPUT + "reports/mapping_percentage/{cid}.{rid}.txt", cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + "reports/mapping_percentage.csv"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "pore_c"
    shell:
        """python scripts/compile_mapping_percent.py {output} {input}"""
        
        
rule pairs_reports:
    input:
        expand(OUTPUT + "pairs/{cid}.{rid}.{{kind}}.pairs", cid=cell_ids, rid=ref_ids),
    output:
        basic=OUTPUT + "reports/pairs/{kind}.basic.csv",
        reads=OUTPUT + "reports/pairs/{kind}.reads.csv",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='expanded|direct',
    conda:
        "bioinf"
    shell:
        """python scripts/pairs_reports.py {output.basic} {output.reads} {input}"""


rule compile_annotate_summaries:
    input:
        expand(OUTPUT + "annotate/{cid}.{rid}.summary.json", cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + "reports/annotate/cardinality.csv",
        OUTPUT + "reports/annotate/pair_types.csv",
        OUTPUT + "reports/annotate/cis_trans.csv",
    params:
        prefix=OUTPUT + "reports/annotate/",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "bioinf"
    shell:
        """python scripts/compile_annotate_summaries.py {params.prefix} {input}"""
       

rule merge_pairtools_stats:
    input:
        expand(OUTPUT + 'reports/pairtools/{{kind}}/{cid}.{rid}.stats.txt', cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + "reports/pairtools/merged.{kind}.stats"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='expanded|direct',
    conda:
        "pairtools"
    shell:
        """pairtools stats -o {output} --merge {input}"""


