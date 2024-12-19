import os
import sys
import pandas as pd
import numpy as np
import glob
import time
import gget
import scipy
from scipy.sparse import csr_matrix
import anndata as an
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from importlib import reload
import warnings
import ot

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


def extract_chromosome(adata, chromosome, order_threshold=1, min_read_count=1):
    """
    Filters an AnnData object based on chromosome, order threshold, and read count.

    Args:
        adata: AnnData object to filter.
        chromosome: Chromosome to select.
        order_threshold: Minimum order (sum of reads across cells) for a gene to be kept.
        min_read_count: Minimum read count for a gene to be kept.

    Returns:
        Filtered AnnData object.
    """
    mask = (adata.obs['chrom'] == chromosome)
    cdata = adata[mask,]
    cdata = cdata[:, cdata.X.sum(axis=0) > min_read_count]

    # recompute the order
    cdata.var['chrom_order'] = np.ravel(cdata.X.sum(axis=0))
    cdata = cdata[:, cdata.var['chrom_order'] > order_threshold]

    # recompute the reads
    cdata.obs['chrom_degree'] = np.ravel(cdata.X.sum(axis=1))

    # Sort the entire AnnData object (cdata) by 'chrom_bin'
    cdata = cdata[cdata.obs.sort_values('chrom_bin').index, :]

    return cdata