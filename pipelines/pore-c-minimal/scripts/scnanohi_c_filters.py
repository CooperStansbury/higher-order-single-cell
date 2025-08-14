import pandas as pd
from pyranges import PyRanges

def filter_adjacent_fragments(cont_df):
    """Filters out adjacent fragments."""
    return cont_df[~cont_df["contact_fragment_adjacent"]].copy()

def filter_close_contacts(cont_df):
    """Filters out close contacts (cis and < 1000bp)."""
    return cont_df[
        ~(
            (cont_df["contact_is_cis"])
            & (cont_df["contact_genome_distance"].abs() < 1000)
        )
    ].copy()

def filter_duplicate_contacts(cont_df):
    """Filters out duplicate contacts."""
    read_stat = (
        cont_df["read_name"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "read_name", "read_name": "n_contacts"})
    )
    cont_df = cont_df.merge(read_stat, on="read_name", how="left")
    cont_df = cont_df.sort_values(by="n_contacts", ascending=False)
    return cont_df.drop_duplicates(
        subset=["align1_fragment_id", "align2_fragment_id"]
    )

def filter_promiscuous_fragments(cont_df):
    """Filters out promiscuous fragments."""
    frag_df = (
        pd.concat([cont_df["align1_fragment_id"], cont_df["align2_fragment_id"]])
        .value_counts()
        .reset_index()
        .rename(columns={"index": "fragment_id", 0: "Freq"})
    )
    prom_fg = frag_df[frag_df["Freq"] > 10]["fragment_id"].tolist()
    return cont_df[
        ~(
            cont_df["align1_fragment_id"].isin(prom_fg)
            | cont_df["align2_fragment_id"].isin(prom_fg)
        )
    ]

def filter_isolated_fragments(cont_df):
    """Filters out isolated fragments."""
    frag_gr_df = pd.concat(
        [
            cont_df[
                ["align1_fragment_id", "align1_chrom", "align1_fragment_start", "align1_fragment_end"]
            ].rename(
                columns={
                    "align1_fragment_id": "id",
                    "align1_chrom": "Chromosome",
                    "align1_fragment_start": "Start",
                    "align1_fragment_end": "End",
                }
            ),
            cont_df[
                ["align2_fragment_id", "align2_chrom", "align2_fragment_start", "align2_fragment_end"]
            ].rename(
                columns={
                    "align2_fragment_id": "id",
                    "align2_chrom": "Chromosome",
                    "align2_fragment_start": "Start",
                    "align2_fragment_end": "End",
                }
            ),
        ]
    ).drop_duplicates()

    frag_gr = PyRanges(frag_gr_df)
    dis = frag_gr.nearest()  
    iso_frag = frag_gr[dis.df["Distance"] > 10000000].df["id"].tolist()

    return cont_df[
        ~(
            cont_df["align1_fragment_id"].isin(iso_frag)
            | cont_df["align2_fragment_id"].isin(iso_frag)
        )
    ]


def contact_filter(cont_df):
    """Main filtering function coordinating the individual filters."""

    # Add index and initial filter column
    cont_df["idx"] = range(len(cont_df))
    cont_df["filter"] = "pass"

    # Apply filters sequentially
    cont_df_filtered = cont_df.copy()
    for filter_func, filter_name in [
        (filter_adjacent_fragments, "adjacent"),
        (filter_close_contacts, "close"),
        (filter_duplicate_contacts, "duplication"),
        (filter_promiscuous_fragments, "promiscuous"),
        (filter_isolated_fragments, "isolated"),
    ]:
        filtered_df = filter_func(cont_df_filtered)
        cont_df.loc[
            (cont_df["filter"] == "pass")
            & (~cont_df["idx"].isin(filtered_df["idx"])),
            "filter",
        ] = filter_name
        cont_df_filtered = filtered_df.copy() 

    return cont_df
