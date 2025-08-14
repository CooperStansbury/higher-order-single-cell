import os
import sys
import pandas as pd
import numpy as np
import pyranges as pr
import pyBigWig
import glob
import time
from scipy.sparse import csr_matrix
import anndata as an
import scanpy as sc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        # logging.FileHandler("process_data.log"),  # Uncomment to also log to a file
    ]
)

if __name__ == "__main__":
    logging.info("Starting script...")

    chrom_sizes_path = sys.argv[1]
    logging.info(f"Chromosome sizes path: {chrom_sizes_path}")
    fragments_path = sys.argv[2]
    logging.info(f"Restriction fragments path: {fragments_path}")
    gene_table_path = sys.argv[3]
    logging.info(f"Gene table path: {gene_table_path}")
    expression_path = sys.argv[4]
    logging.info(f"Expression data path: {expression_path}")
    feature_path = sys.argv[5]
    logging.info(f"Feature path: {feature_path}")
    output_path = sys.argv[6]
    logging.info(f"Output path: {output_path}")

    # load the chrom sizes
    logging.info("Loading chromosome sizes...")
    try:
        cdf = pd.read_csv(chrom_sizes_path)
        logging.info(f"Loaded chromosome sizes. Shape: {cdf.shape}")
        logging.debug(f"Chromosome sizes head:\n{cdf.head()}")
    except Exception as e:
        logging.error(f"Error loading chromosome sizes: {e}")
        sys.exit(1)  # Exit on critical error

    # load the restriction fragments
    logging.info("Loading restriction fragments...")
    try:
        df = pd.read_parquet(fragments_path)
        logging.info(f"Loaded restriction fragments. Shape: {df.shape}")
        logging.debug(f"Restriction fragments head:\n{df.head()}")
        df = df[df['Chromosome'].isin(cdf['chrom'])]
        logging.info(f"Filtered restriction fragments by chromosomes in chrom sizes. New shape: {df.shape}")
        del cdf
        logging.info("Deleted chromosome sizes dataframe.")
    except Exception as e:
        logging.error(f"Error loading restriction fragments: {e}")
        sys.exit(1)


    # load the gene annotations
    logging.info("Loading gene annotations...")
    try:
        gdf = pd.read_parquet(gene_table_path)
        logging.info(f"Loaded gene annotations. Shape: {gdf.shape}")
        logging.debug(f"Gene annotations head:\n{gdf.head()}")
    except Exception as e:
        logging.error(f"Error loading gene annotations: {e}")
        sys.exit(1)

    # load the expression data
    logging.info("Loading expression data...")
    try:
        edf = pd.read_parquet(expression_path, columns=['ens_gene_id', 'TPM'])
        logging.info(f"Loaded expression data. Shape: {edf.shape}")
        logging.debug(f"Expression data head:\n{edf.head()}")
        edf.columns = ['gene_id', 'TPM']
        logging.info("Renamed expression data columns.")
        gdf = pd.merge(gdf, edf, how='left')
        logging.info(f"Merged gene annotations with expression data. New shape: {gdf.shape}")
        del edf
        logging.info("Deleted raw expression data.")
    except Exception as e:
        logging.error(f"Error loading/merging expression data: {e}")
        sys.exit(1)


    """ MERGE GENES AND RESTRICTION FRAGMENTS """
    logging.info("Merging genes and restriction fragments...")
    # Convert to PyRanges objects
    pr_df = pr.PyRanges(df)
    logging.info("Converted restriction fragments to PyRanges object.")
    pr_gdf = pr.PyRanges(gdf)
    logging.info("Converted gene annotations to PyRanges object.")

    # Perform the join.
    try:
        merged_pr = pr_df.join(
            pr_gdf,
            how="left",
            suffix="_gene",
            slack=500,
            report_overlap=True,
            preserve_order=True,
        )  # left join keeps all rows from df
        logging.info("Performed join operation between restriction fragments and gene annotations.")
    except Exception as e:
        logging.error(f"Error during PyRanges join: {e}")
        sys.exit(1)

    # Convert back to pandas DataFrame
    merged_df = merged_pr.df
    logging.info("Converted merged PyRanges object back to pandas DataFrame.")
    del pr_gdf
    logging.info("Deleted gene annotations PyRanges object.")
    del pr_df
    logging.info("Deleted restriction fragments PyRanges object.")
    del merged_pr
    logging.info("Deleted merged PyRanges object.")

    # Select the best overlap for each interval
    merged_df = merged_df.sort_values(by=['Chromosome', 'Start', 'Overlap'], ascending=[True, True, False])
    logging.info("Sorted merged DataFrame by Chromosome, Start, and Overlap.")
    merged_df = merged_df.drop_duplicates(subset=['fragment_id',], keep='first')
    logging.info("Dropped duplicate fragment_ids, keeping the first (best overlap).")
    try:
        assert(df.shape[0] == merged_df.shape[0])  # there cannot be added rows!
        logging.info("Assertion passed: Number of rows remains the same after merging and dropping duplicates.")
    except AssertionError as e:
        logging.error(f"Assertion failed: Row count mismatch after merge: {e}")
        sys.exit(1)
    logging.info(f"Merged dataframe shape: {merged_df.shape}")


    """ MERGE THE FEATURES """
    logging.info("Merging features...")

    # load the feature paths
    try:
        feature_paths = pd.read_csv(feature_path)
        logging.info(f"Loaded feature paths. Shape: {feature_paths.shape}")
        logging.debug(f"Feature paths head:\n{feature_paths.head()}")
    except Exception as e:
        logging.error(f"Error loading feature paths: {e}")
        sys.exit(1)

    wigs = {}
    for _, feature_row in feature_paths.iterrows():
        file_id = feature_row['file_id']
        file_path = feature_row['file_path']
        logging.info(f"Opening BigWig file: {file_path} (ID: {file_id})")
        try:
            bigwig = pyBigWig.open(file_path)
            wigs[file_id] = bigwig
        except Exception as e:
            logging.error(f"Error opening BigWig file {file_path}: {e}")
            # Don't exit here, try to continue with other files
    logging.info(f"Loaded {len(wigs)} BigWig files.")

    # actually merge the data
    feature_rows = []
    logging.info("Iterating through merged DataFrame to extract features...")
    start_time = time.time()
    for idx, row in merged_df.iterrows():
        chrom = row['Chromosome']
        start = row['Start']
        end = row['End']
        new_row = {'index' : idx}
        for k, bw in wigs.items():
            try:
                value = bw.stats(f"chr{chrom}", start, end, type='mean')[0]
            except Exception as e:
                logging.warning(f"Error retrieving stats for Chromosome: {chrom}, Start: {start}, End: {end}, BigWig ID: {k}. Error: {e}. Setting value to NA.")
                value = pd.NA

            new_row[k] = value
        feature_rows.append(new_row)
    end_time = time.time()
    logging.info(f"Finished iterating. Time taken: {end_time - start_time:.2f} seconds")
    logging.info(f"Collected features for {len(feature_rows)} rows.")

    features = pd.DataFrame(feature_rows)
    logging.info(f"Created features DataFrame. Shape: {features.shape}")
    features = features.set_index('index')
    logging.info("Set 'index' column as index in features DataFrame.")
    logging.debug(f"Features head:\n{features.head()}")

    # combine everything into a huge dataframe
    merged_df = pd.concat([merged_df, features], ignore_index=False, axis=1)
    logging.info(f"Concatenated original DataFrame with features DataFrame. Final shape: {df.shape}")

    del features
    logging.info("Deleted features DataFrame.")

    for col in merged_df.columns:
        if merged_df[col].dtype == 'object':
            try:
                merged_df[col] = merged_df[col].astype(str)
                logging.info(f"Converted column '{col}' to string.")
            except Exception as e: # Catch a more specific exception if possible
                logging.warning(f"Column '{col}' is of type 'object', but not converted (likely not boolean).  Error: {e}")
        elif merged_df[col].dtype == 'bool':
            merged_df[col] = merged_df[col].astype(str)
            logging.info(f"Converted column '{col}' from bool to string.")
        else:
            logging.debug(f"Column '{col}' is of type '{merged_df[col].dtype}' and not converted.")  # Use debug level for this

    # write the output
    logging.info(f"Writing final DataFrame to: {output_path}")
    try:
        merged_df.to_parquet(output_path)
    except Exception as e:
        logging.error(f"Error writing output to parquet: {e}")
        sys.exit(1)
    logging.info("Script finished successfully.")

