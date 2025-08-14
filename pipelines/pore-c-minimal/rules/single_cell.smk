rule li_et_al_filters:
    input:
        OUTPUT + "pairs/{cid}.{rid}.{kind}.pairs",
    output:
        pairs=OUTPUT + "single_cell/{kind}/{cid}.{rid}.filtered.pairs",
        stats=OUTPUT + "reports/single_cell/filters/{cid}.{rid}.{kind}.csv",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='expanded|direct',
    conda:
        "bioinf"
    shell:
        """python scripts/single_cell_contact_filter.py {input} {output.pairs} {output.stats}"""
        
        

rule annotate_duplicates:
    input:
        OUTPUT + "align_table/{cid}.{rid}.align_table.parquet",
    output:
        OUTPUT + "single_cell/duplication_reports/{cid}.{rid}.dupes.parquet"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "bioinf"
    shell:
        """python scripts/detect_duplicates.py {input} {output}"""
        
        
rule duplication_report:
    input:
        expand(OUTPUT + "single_cell/duplication_reports/{cid}.{rid}.dupes.parquet", cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + "reports/single_cell/duplication_report.csv"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "bioinf"
    shell:
        """python scripts/compile_duplication_report.py {output} {input}"""
    
        
        
rule pairs_reports_single_cell:
    input:
        expand(OUTPUT + "single_cell/{{kind}}/{cid}.{rid}.filtered.pairs", cid=cell_ids, rid=ref_ids),
    output:
        basic=OUTPUT + "reports/single_cell/pairs/{kind}.basic.csv",
        reads=OUTPUT + "reports/single_cell/pairs/{kind}.reads.csv",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='expanded|direct',
    conda:
        "bioinf"
    shell:
        """python scripts/pairs_reports.py {output.basic} {output.reads} {input}"""

        
        
rule compile_filter_reports:
    input:
        expand(OUTPUT + "reports/single_cell/filters/{cid}.{rid}.{{kind}}.csv", cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + "reports/single_cell/{kind}.filter_report.csv"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='direct|expanded',
    conda:
        "bioinf"
    shell:
        """python scripts/compile_single_cell_filter.py {output} {input}"""
        
        
        
rule single_cell_pairstats:
    input:
        OUTPUT + "single_cell/{kind}/{cid}.{rid}.filtered.pairs",
    output:
        OUTPUT + "reports/single_cell/pairtools/{kind}/{cid}.{rid}.stats.txt"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='direct|expanded',
    conda:
        "pairtools"
    shell:
        """pairtools stats {input} -o {output}"""
        
        
rule single_cell_pairstats_merged:
    input:
        expand(OUTPUT + 'reports/single_cell/pairtools/{{kind}}/{cid}.{rid}.stats.txt', cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + "reports/single_cell/pairtools/{kind}.stats"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='expanded|direct',
    conda:
        "pairtools"
    shell:
        """pairtools stats -o {output} --merge {input}"""

    