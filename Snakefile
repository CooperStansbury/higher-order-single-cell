OUTPUT = "/scratch/indikar_root/indikar1/shared_data/higher_order/"


rule all:
    input:
        
        
rule make_schic_ref:
    input:
        ref="/nfs/turbo/umms-indikar/shared/projects/poreC/data/nagano2017/schic2_mm9/seq/redb/GATC.fends",
        res="config/resolutions.txt"
    output:
        f"{OUTPUT}schic_resources/reference.csv",
    shell:
        """python scripts/make_schic_ref.py {input.ref} {input.res} {output}"""