import os
import sys
import pandas as pd
import numpy as np
import glob
import time
from scipy.sparse import csr_matrix
import anndata as an
import scanpy as sc
import psutil


def generate_report(anndata_path):
    """
    Generates a formatted report summarizing key metrics from an AnnData object.

    Args:
        anndata_path (str): Path to the AnnData file (.h5ad).
    """

    adata = sc.read_h5ad(anndata_path)

    n, m = adata.shape
    n_samples = adata.var['basename'].nunique()
    contacts_per_sample = m / n_samples
    mean_order = adata.var['order'].mean()
    median_order = adata.var['order'].median()
    adata.var['higher_order'] = adata.var['order'] > 2
    coverage = adata.X.shape[0] / adata.uns['intervals'].shape[0]
    resolution = adata.uns['base_resolution']

    # Additional Metrics
    sparsity = 1 - (adata.X.nnz / (adata.shape[0] * adata.shape[1]))  # Calculate sparsity
    mean_reads_per_cell = adata.X.sum(axis=1).mean()  # Calculate mean reads per cell

    mean_reads_per_bin = adata.X.sum(axis=0).mean()  # Sum across rows (cells)

    mean_genes_per_cell = adata.obs['n_genes'].mean()
    median_genes_per_cell = adata.obs['n_genes'].median()

    mean_genes_per_contact = adata.var['n_genes'].mean()
    median_genes_per_contact = adata.var['n_genes'].median()

    chromosome_distribution = adata.obs['chrom'].value_counts(normalize=True)

    # Report Formatting
    print("-" * 30)
    print("AnnData Summary Report")
    print("-" * 30)
    print(f"File Path: {anndata_path}")
    print("-" * 30)
    print(f"Number of rows (cells): {n}")
    print(f"Number of columns (contacts): {m}")
    print(f"Number of samples: {n_samples}")
    print(f"Average number of contacts per sample: {contacts_per_sample:.4f}")
    print(f"Average order: {mean_order:.4f}")
    print(f"Median order: {median_order:.4f}")
    print(f"Matrix sparsity: {sparsity:.4f}")
    print(f"Mean reads per cell: {mean_reads_per_cell:.4f}")
    print(f"Mean reads per bin: {mean_reads_per_bin:.4f}")
    print("-" * 30)
    print("Proportion of Contacts:")
    print(adata.var['higher_order'].value_counts(normalize=True))
    print("-" * 30)
    print(f"Coverage: {coverage:.4f} of loci at {resolution}bp resolution")
    print("-" * 30)
    print("Gene Metrics:")
    print(f"Mean genes per cell: {mean_genes_per_cell:.4f}")
    print(f"Median genes per cell: {median_genes_per_cell:.4f}")
    print(f"Mean genes per contact: {mean_genes_per_contact:.4f}")
    print(f"Median genes per contact: {median_genes_per_contact:.4f}")
    print("-" * 30)
    print("Chromosome Distribution:")
    print(chromosome_distribution) 
    print("-" * 30)

if __name__ == "__main__":
    anndata_path = sys.argv[1]
    generate_report(anndata_path)