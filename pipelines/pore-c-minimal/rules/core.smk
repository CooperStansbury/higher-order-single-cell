rule copy_fastq:
    input:
        fastq_df['file_path'].to_list()
    output:
        fastq_df['out_path'].to_list(),
    run:
        from shutil import copyfile
        for i, fpath in enumerate(input):

            outPath = output[i]
            copyfile(fpath, outPath)
            
            
rule fastq_raw_report:
    input:
        expand(OUTPUT + "fastq/{cid}.raw.fastq", cid=cell_ids),
    output:
        OUTPUT + "reports/seqkit/fastq.report.txt",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
    threads:
        config['threads'] // 4
    conda:
        "bioinf"
    shell:
        """seqkit stats -a -b -j {threads} {input} -o {output}"""


rule minimap_align:
    input:
        fastq=OUTPUT + "fastq/{cid}.raw.fastq",
        ref=OUTPUT + 'references/{rid}.fa',
        ind=OUTPUT + 'references/{rid}.mmi',
    output:
        bam=OUTPUT + 'alignments/{cid}.{rid}.bam'
    threads:
        config['threads'] // 4
    params:
        args=config['minimap_args'],
        temp=OUTPUT,
    conda:
        "aligner"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    shell:
        """minimap2 {params.args} -t {threads} {input.ref} {input.fastq} \
        | samtools sort -O bam -T {params.temp} -o {output.bam} """


rule align_digest:
    input:
        fastq=OUTPUT + 'fastq/{cid}.raw.fastq',
        bam=OUTPUT + 'alignments/{cid}.{rid}.bam',
        ref=OUTPUT + 'references/{rid}.fa',
    output:
        OUTPUT + 'digest/{cid}.{rid}.bam'
    threads:
        config['threads'] // 4
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    params:
        enzyme=config['enzyme']
    conda:
        "pore_c"
    shell:
        """scripts/align-digest.sh -f {input.fastq} \
        -b {input.bam} -r {input.ref} -e {params.enzyme} \
        -t {threads} -o {output}"""


rule annotate:
    input:
        bam=OUTPUT + 'digest/{cid}.{rid}.bam'
    output:
        bam=OUTPUT + 'annotate/{cid}.{rid}.ns.bam',
    threads:
        config['threads'] // 4
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    params:
        prefix=lambda wildcards: OUTPUT + "annotate/" + wildcards.cid + "." + wildcards.rid
    conda:
        "pore_c"
    shell:
        """scripts/annotate.sh -b {input} -o {params.prefix} -- --threads {threads} --direct_only"""



rule make_align_table:
    input:
        bam=OUTPUT + "annotate/{cid}.{rid}.ns.bam",
        db=OUTPUT + "references/{rid}.fragment_db.parquet",
    output:
        OUTPUT + "align_table/{cid}.{rid}.align_table.parquet",
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
    conda:
        "pore_c"
    shell:
        """python scripts/bam2align_table.py {input.bam} {input.db} {output}"""
        

rule pairs_direct:
     input:
         bam=OUTPUT + 'annotate/{cid}.{rid}.ns.bam', # must be the monomer file
         sizes=OUTPUT + 'references/{rid}.chrom.sizes',
         fragments=OUTPUT + 'references/{rid}.fragments.bed',
     output:
         pairs=OUTPUT + "pairs/{cid}.{rid}.direct.pairs",
         stats=OUTPUT + "reports/pairtools/direct/{cid}.{rid}.stats.txt",
     wildcard_constraints:
         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
     conda:
         "pairtools"
     params:
         columns="mapq,read_len,algn_read_span,algn_ref_span,matched_bp"
     shell:
         """pairtools parse2 --output-stats {output.stats} \
         --chroms-path {input.sizes} \
         --drop-sam \
         --drop-seq \
         --add-pair-index \
         --single-end \
         --no-expand \
         --no-flip \
         --report-position 'walk' \
         --report-orientation 'walk' \
         --readid-transform 'readID.split(":")[0]' \
         --add-columns {params.columns} \
         {input.bam} | pairtools restrict  -f {input.fragments} -o {output.pairs}"""



rule pairs_expanded:
     input:
         bam=OUTPUT + 'annotate/{cid}.{rid}.pe.bam', # must be the paired-end file
         sizes=OUTPUT + 'references/{rid}.chrom.sizes',
         fragments=OUTPUT + 'references/{rid}.fragments.bed',
     output:
         pairs=OUTPUT + "pairs/{cid}.{rid}.expanded.pairs",
         stats=OUTPUT + "reports/pairtools/expanded/{cid}.{rid}.stats.txt",
     wildcard_constraints:
         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
     conda:
         "pairtools"
     params:
         columns="mapq,read_len,algn_read_span,algn_ref_span,matched_bp"
     shell:
         """pairtools parse2 --output-stats {output.stats} \
         --chroms-path {input.sizes} \
         --drop-sam \
         --drop-seq \
         --expand \
         --add-pair-index \
         --no-flip \
         --report-position 'walk' \
         --report-orientation 'walk' \
         --readid-transform 'readID.split(":")[0]' \
         --add-columns {params.columns} \
         {input.bam} | pairtools select "walk_pair_type.__contains__('E')" | \
         pairtools restrict  -f {input.fragments} -o {output.pairs}"""






  
# rule pairs_expanded:
#     input:
#         bam=OUTPUT + 'expanded/annotate/{cid}.{rid}.pe.bam',
#         sizes=OUTPUT + 'references/{rid}.chrom.sizes',
#         fragments=OUTPUT + 'references/{rid}.fragments.bed',
#     output:
#         pairs=OUTPUT + "expanded/pairs/{cid}.{rid}.expanded.pairs",
#         stats=OUTPUT + "reports/pairtools/expanded/{cid}.{rid}.stats.txt",
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     conda:
#         "pairtools"
#     params:
#         columns="mapq,read_len,algn_read_span,algn_ref_span,matched_bp"
#     shell:
#         """pairtools parse2 --output-stats {output.stats} \
#         --chroms-path {input.sizes} \
#         --drop-sam \
#         --drop-seq \
#         --expand \
#         --single-end \
#         --report-position 'junction' \
#         --report-orientation 'read' \
#         --readid-transform 'readID.split(":")[0]' \
#         --add-columns {params.columns} \
#         {input.bam} | pairtools restrict  -f {input.fragments} -o {output.pairs}"""
#         
#         
# rule pairs_direct:
#     input:
#         bam=OUTPUT + 'direct/annotate/{cid}.{rid}.pe.bam',
#         sizes=OUTPUT + 'references/{rid}.chrom.sizes',
#         fragments=OUTPUT + 'references/{rid}.fragments.bed',
#     output:
#         pairs=OUTPUT + "direct/pairs/{cid}.{rid}.direct.pairs",
#         stats=OUTPUT + "reports/pairtools/direct/{cid}.{rid}.stats.txt",
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     conda:
#         "pairtools"
#     params:
#         columns="mapq,read_len,algn_read_span,algn_ref_span,matched_bp"
#     shell:
#         """pairtools parse2 --output-stats {output.stats} \
#         --chroms-path {input.sizes} \
#         --drop-sam \
#         --drop-seq \
#         --no-expand \
#         --single-end \
#         --report-position 'junction' \
#         --report-orientation 'read' \
#         --readid-transform 'readID.split(":")[0]' \
#         --add-columns {params.columns} \
#         {input.bam} | pairtools restrict  -f {input.fragments} -o {output.pairs}"""

