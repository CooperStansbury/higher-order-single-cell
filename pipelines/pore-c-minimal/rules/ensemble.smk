rule pooled_pairs:
    input:
        expand(OUTPUT + "pairs/{cid}.{rid}.{{kind}}.pairs", cid=cell_ids, rid=ref_ids),
    output:
        OUTPUT + "ensemble/{kind}.pairs"
    wildcard_constraints:
        cid='|'.join([re.escape(x) for x in set(cell_ids)]),
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='expanded|direct',
    conda:
        "pairtools"
    params:
        tmp=OUTPUT,
        mem='20G',
    shell:
        """pairtools merge --memory {params.mem} --tmpdir {params.tmp} {input} | pairtools sort -o {output}"""


rule pairtools_dedup:
    input:
        OUTPUT + "ensemble/{kind}.pairs"
    output:
        pairs=OUTPUT + "ensemble/{kind}.dedup.pairs", 
        stats=OUTPUT + "reports/pairtools_dedup/{kind}.stats",
    wildcard_constraints:
        kind='expanded|direct',
    conda:
        "pairtools"
    shell:
        """pairtools select 'mapq1>0 and mapq2>0' {input} | pairtools dedup --max-mismatch 3 --mark-dups \
               --output {output.pairs} \
               --output-stats {output.stats}"""


rule pairs_to_cooler:
    input:
        pairs=OUTPUT + "ensemble/{kind}.dedup.pairs", 
        sizes=OUTPUT + "references/{rid}.chrom.sizes", 
    output:
        OUTPUT + "cooler/{kind}.{rid}.cool", 
    wildcard_constraints:
        rid='|'.join([re.escape(x) for x in set(ref_ids)]),
        kind='expanded|direct',
    conda:
        "cooler"
    params:
        base="10000",
    shell:
        """cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 \
         {input.sizes}:{params.base} \
         {input.pairs} {output}"""
         
         
rule cooler_prepare:
    input:
        OUTPUT + "cooler/{kind}.{rid}.cool", 
    output:
        OUTPUT + "cooler/{kind}.{rid}.mcool", 
    conda:
        "cooler"
    params:
        bins="100000,100000",
    threads: 
        config['threads'] // 4
    shell:
        """cooler zoomify \
                --nproc {threads} \
                --out {output} \
                --resolutions {params.bins} \
                --balance {input} """
    
    
# rule expanded_to_cooler:
#     input:
#         pairs=OUTPUT + "expanded_pairs/{cid}.{rid}.pairs.gz",
#         sizes=OUTPUT + 'references/{rid}.chrom.sizes',
#     output:
#         OUTPUT + "cooler_expanded/{cid}.{rid}.cool"
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     params:
#         base=int(config['base_resolution'])
#     shell:
#         """cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 \
#         {input.sizes}:{params.base} \
#         {input.pairs} {output} """
# 
# 
# rule zoom_expanded:
#     input:
#         OUTPUT + "cooler_expanded/{cid}.{rid}.cool"
#     output:
#         OUTPUT + "cooler_expanded/{cid}.{rid}.mcool"
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     params:
#         resolutions=",".join(map(str, config["resolutions"])),
#     shell:
#         """cooler zoomify -r {params.resolutions} \
#         -o {output} {input}"""
# 
# 
# rule merge_expanded:
#     input:
#         expand(OUTPUT + 'cooler_expanded/{cid}.{{rid}}.cool', cid=cell_ids,),
#     output:
#         OUTPUT + "merged_cooler/{rid}.merged.expanded.cool"
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     shell:
#         """cooler merge {output} {input}"""
# 
# 
# rule zoom_merged_expanded:
#     input:
#         cool=OUTPUT + "merged_cooler/{rid}.merged.expanded.cool",
#     output:
#         OUTPUT + "merged_cooler/{rid}.merged.expanded.mcool"
#     wildcard_constraints:
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     params:
#         resolutions=",".join(map(str, config["resolutions"])),
#     shell:
#         """cooler zoomify -r {params.resolutions} \
#         -o {output} {input.cool}"""
# 
# 
# rule balance_merged_expanded:
#     input:
#         OUTPUT + "merged_cooler/{rid}.merged.expanded.mcool"
#     output:
#         touch(OUTPUT + "merged_cooler/{rid}.balance.expanded.done")
#     wildcard_constraints:
#         rid='|'.join([re.escape(x) for x in set(ref_ids)])
#     threads:
#         config['threads'] // 2
#     shell:
#         """cooler ls {input} | xargs -n1 cooler balance -p {threads} -f """
# 
# 
# rule build_pairs_adjacent:
#     input:
#         bam=OUTPUT + 'annotate/{cid}.{rid}.ns.bam',
#         sizes=OUTPUT + 'references/{rid}.chrom.sizes',
#     output:
#         pairs=OUTPUT + "adjacent_pairs/{cid}.{rid}.pairs",
#         stats=OUTPUT + "adjacent_pairs/{cid}.{rid}.pair_stats.txt",
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     params:
#         transform="'" + str(config['readid_transform']) + "'",
#         mapq=config['min_mapq']
#     shell:
#         """pairtools parse2 --output-stats {output.stats} \
#         -c {input.sizes} --drop-sam --drop-seq --no-expand --add-pair-index --readid-transform {params.transform} \
#         --add-columns mapq,read_len,algn_ref_span,algn_read_span {input.bam} > {output.pairs}"""
# 
# # --no-expand
# 
# 
# rule restrict_adjacent:
#     input:
#         pairs=OUTPUT + "adjacent_pairs/{cid}.{rid}.pairs",
#         frags=OUTPUT + 'references/{rid}.fragments.bed',
#     output:
#         OUTPUT + "adjacent_pairs/{cid}.{rid}.pairs.gz"
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     shell:
#         """pairtools restrict -f {input.frags} -o {output} {input.pairs}"""
# 
# 
# rule adjacent_to_cooler:
#     input:
#         pairs=OUTPUT + "adjacent_pairs/{cid}.{rid}.pairs.gz",
#         sizes=OUTPUT + 'references/{rid}.chrom.sizes',
#     output:
#         OUTPUT + "cooler_adjacent/{cid}.{rid}.cool"
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     params:
#         base=int(config['base_resolution'])
#     shell:
#         """cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 \
#         {input.sizes}:{params.base} \
#         {input.pairs} {output} """
# 
# 
# rule zoom_adjacent:
#     input:
#         OUTPUT + "cooler_adjacent/{cid}.{rid}.cool"
#     output:
#         OUTPUT + "cooler_adjacent/{cid}.{rid}.mcool"
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     params:
#         resolutions=",".join(map(str, config["resolutions"])),
#     shell:
#         """cooler zoomify -r {params.resolutions} \
#         -o {output} {input}"""
# 
# 
# rule merge_adjacent:
#     input:
#         expand(OUTPUT + 'cooler_adjacent/{cid}.{{rid}}.cool', cid=cell_ids,),
#     output:
#         OUTPUT + "merged_cooler/{rid}.merged.adjacent.cool"
#     wildcard_constraints:
#         cid='|'.join([re.escape(x) for x in set(cell_ids)]),
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     shell:
#         """cooler merge {output} {input}"""
# 
# 
# rule zoom_merged_adjacent:
#     input:
#         cool=OUTPUT + "merged_cooler/{rid}.merged.adjacent.cool",
#     output:
#         OUTPUT + "merged_cooler/{rid}.merged.adjacent.mcool"
#     wildcard_constraints:
#         rid='|'.join([re.escape(x) for x in set(ref_ids)]),
#     params:
#         resolutions=",".join(map(str, config["resolutions"])),
#     shell:
#         """cooler zoomify -r {params.resolutions} \
#         -o {output} {input.cool}"""
# 
# 
# rule balance_merged_adjacent:
#     input:
#         OUTPUT + "merged_cooler/{rid}.merged.adjacent.mcool"
#     output:
#         touch(OUTPUT + "merged_cooler/{rid}.balance.adjacent.done")
#     wildcard_constraints:
#         rid='|'.join([re.escape(x) for x in set(ref_ids)])
#     threads:
#         config['threads'] // 2
#     shell:
#         """cooler ls {input} | xargs -n1 cooler balance -p {threads} -f """
# 


