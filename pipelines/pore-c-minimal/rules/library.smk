rule merge_alignments:
    input:
        expand(OUTPUT + "alignments/{cid}.{rid}.bam", cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + 'library/merged.bam'
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:       
        "pore_c"
    threads:
        config['threads']
    params:
        temp=OUTPUT
    shell:
        """samtools merge -@ {threads} - {input} | samtools sort -@ {threads} -T {params.temp} -o {output}"""
        
        
rule mark_duplicates:
    input:
         OUTPUT + 'library/merged.bam',
    output:
         bam=OUTPUT + 'library/merged.duplicates.bam',
         report=OUTPUT + 'reports/gatk/duplicates.txt',
    log:
        OUTPUT + 'reports/gatk/duplicates.log'
    conda:
        "gatk" 
    shell:
        """gatk MarkDuplicates I={input} O={output.bam} M={output.report} 2> {log}"""
        
        
rule library_coverage:
    input:
        OUTPUT + 'library/merged.duplicates.bam',
    output:
        OUTPUT + "reports/samtools/library_coverage.txt"
    conda:
        "bioinf"
    shell:
        """samtools coverage {input} -o {output} """
        
        
rule samtools_flagstat:
    input:
        OUTPUT + 'library/merged.duplicates.bam',
    output:
        OUTPUT + "reports/samtools/flagstat.txt"
    conda:
        "bioinf"
    threads:
        config['threads'] // 4 
    shell:
        """samtools flagstat -@ {threads} -O 'tsv' {input} > {output}"""
        
                
rule get_read_lengths:
    input:
        fastq=OUTPUT + "fastq/{cid}.raw.fastq",
    output:
        OUTPUT + "reports/read_lengths/{cid}.read_lengths.parquet"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "bioinf"
    shell:
        """python scripts/get_read_lengths.py {input} {output}"""
        
               