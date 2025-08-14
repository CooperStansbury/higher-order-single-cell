import pandas as pd
import os
from pathlib import Path


def load_pod5_df(pod5_paths, output_path, 
                  output_subdir="pod5/",
                  ext=".pod5"):
    """A function to load the fastq output df from a 
    file """
    pod5_df = pd.read_csv(pod5_paths, comment="#")
    pod5_df['basename'] = pod5_df['file_path'].apply(lambda x: x.split("/")[-1])
    pod5_df['ext'] = ext
    pod5_df['out_dir'] = output_path
    pod5_df['sub_dir'] = output_subdir
    pod5_df['out_path'] = pod5_df['out_dir'] + pod5_df['sub_dir'] \
                        + pod5_df['cell_id'] + pod5_df['ext']
    return pod5_df

        
    
def load_fastq_df(fastq_paths, output_path, 
                  output_subdir="fastq/",
                  ext=".raw.fastq"):
    """A function to load the fastq output df from a 
    file """
    fastq_df = pd.read_csv(fastq_paths, comment="#")
    fastq_df['basename'] = fastq_df['file_path'].apply(lambda x: x.split("/")[-1])
    fastq_df['ext'] = ext
    fastq_df['out_dir'] = output_path
    fastq_df['sub_dir'] = output_subdir
    fastq_df['out_path'] = fastq_df['out_dir'] + fastq_df['sub_dir'] \
                        + fastq_df['cell_id'] + fastq_df['ext']
    return fastq_df


def load_ref_df(ref_paths, output_path,
                output_subdir="references/", ext=".fa"):
    """A function to load the references to a 
    dataframe """
    ref_df = pd.read_csv(ref_paths, comment="#")
    ref_df['basename'] = ref_df['file_path'].apply(lambda x: x.split("/")[-1])
    ref_df['ext'] = ext
    ref_df['out_dir'] = output_path
    ref_df['sub_dir'] = output_subdir
    ref_df['out_path'] = ref_df['out_dir'] + ref_df['sub_dir'] \
                    + ref_df['ref_id'] + ref_df['ext']

    return ref_df


def load_snp_df(snp_paths, output_path,
                output_subdir="vcf/", ext=".vcf.gz"):
    """A function to load the VCFs to a 
    dataframe """
    snp_df = pd.read_csv(snp_paths, comment="#")
    snp_df['basename'] = snp_df['file_path'].apply(lambda x: x.split("/")[-1])
    snp_df['ext'] = ext
    snp_df['out_dir'] = output_path
    snp_df['sub_dir'] = output_subdir
    snp_df['out_path'] = snp_df['out_dir'] + snp_df['sub_dir'] \
                    + snp_df['snp_id'] + snp_df['ext']

    return snp_df


def load_gtf_df(gtf_paths, output_path,
                output_subdir="gtf/", ext=".gtf.gz"):
    """A function to load the GTFs to a 
    dataframe """
    gtf_df = pd.read_csv(gtf_paths, comment="#")
    gtf_df['basename'] = gtf_df['file_path'].apply(lambda x: x.split("/")[-1])
    gtf_df['ext'] = ext
    gtf_df['out_dir'] = output_path
    gtf_df['sub_dir'] = output_subdir
    gtf_df['out_path'] = gtf_df['out_dir'] + gtf_df['sub_dir'] \
                    + gtf_df['gtf_id'] + gtf_df['ext']

    return gtf_df
                    
    
    

    