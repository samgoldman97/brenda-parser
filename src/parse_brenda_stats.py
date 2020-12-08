"""Module to hold statistics functions for post processing of parsed data

"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from rdkit import Chem
from typing import Union

from src import utils

def get_rxn_compound_stats(rxn_set: pd.DataFrame,
                           compound_set: pd.DataFrame) -> dict:
    """get_rxn_compound_stats.

    Args:
        rxn_set (pd.DataFrame): rxn_set
        compound_set (pd.DataFrame): compound_set

    Returns:
        dict:
    """

    stats_summary = dict()
    seq_is_null = rxn_set["SEQ_ID"].apply(lambda x: x == None or x == "" or x == "UNK")
    seq_not_null = ~seq_is_null 
    stats_summary["Avg rxn per enzyme"] = np.mean(
        rxn_set[seq_not_null].groupby("SEQ_ID").apply(len))
    stats_summary["Avg rxn per ec"] = np.mean(
        rxn_set.groupby("EC_NUM").apply(len))
    stats_summary["Total rxn,enzyme pairs with sequences"] = len(
        rxn_set[seq_not_null])
    stats_summary["Total rxn,enzyme pairs with no seqs"] = len(
        rxn_set[seq_is_null])

    stats_summary["Total unique enzymes"] = len(set(rxn_set["SEQ_ID"]) )
    stats_summary["Avg enzymes per class"] = np.mean(rxn_set.groupby("EC_NUM").apply(lambda x : len(set(x["SEQ_ID"]))))

    # Concatenate together with a space after substrates..
    full_rxn_string = (rxn_set["SUBSTRATES"].apply(lambda x: x.strip() + " ") +
                       rxn_set["PRODUCTS"].apply(lambda x: x.strip()))

    rxn_vector = pd.unique(full_rxn_string)
    stats_summary["Total unique rxns"] = len(rxn_vector)
    stats_summary["Total unique rxns without UNK in sub or prod"] = np.sum(
        ["UNK" not in i for i in rxn_vector])

    seq_and_no_unk = np.logical_and(
        seq_not_null, np.array(["UNK" not in i for i in full_rxn_string]))

    stats_summary["No UNK and Seq entries"] = np.sum(seq_and_no_unk)
    stats_summary["Total unique compounds"] = len(
        set([j for i in rxn_vector for j in i.split()]))
    stats_summary["Avg rxn, enzyme pairs in each EC class"] = np.mean(
        rxn_set[seq_not_null].groupby("EC_NUM").apply(len))
    stats_summary["Num EC Classses with more than 10 enz, rxn pairs"] = np.sum(
        rxn_set[seq_not_null].groupby("EC_NUM").apply(lambda x: len(x) > 10))
    stats_summary["Num EC Classses with at least 1 enz, rxn pairs"] = np.sum(
        rxn_set[seq_not_null].groupby("EC_NUM").apply(lambda x: len(x) > 1))

    stats_summary["Num EC Classses with subs"] = len(set(rxn_set["EC_NUM"].values))

    if compound_set is not None: 
        inhibitor_entries = compound_set[compound_set["COMPOUND_TYPE"] ==
                                         "INHIBITORS"]
        activator_entries = compound_set[compound_set["COMPOUND_TYPE"] ==
                                         "ACTIVATING_COMPOUND"]
        stats_summary["Unique inhibitors"] = len(
            pd.unique(inhibitor_entries["COMPOUND"]))
        stats_summary["Unique activators"] = len(
            pd.unique(activator_entries["COMPOUND"]))
        stats_summary["Avg inhibitors per EC"] = np.mean(
            inhibitor_entries.groupby("EC_NUM").apply(len))
        stats_summary["Avg activators per EC"] = np.mean(
            activator_entries.groupby("EC_NUM").apply(len))
    return stats_summary

def plot_unique_elements(df : pd.DataFrame, mol_col : str, 
                         groupby_col : str, split_stack_mol : bool =True, 
                         split_stack_group : bool = True):
    """Plot unique (space split) elemnets of df in col mol_col groupbyed by groupby_col"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context="paper", font_scale=2, rc = {"figure.figsize" : (20,10)})

    df_new = df.copy()

    # Drop NA rows to avoid counting issues
    df_new = df_new.dropna(subset=[mol_col, groupby_col]).reset_index(drop=True)

    if split_stack_mol: 
        # Split all the col of interest 
        df_new = pd.DataFrame(df_new[mol_col].str.split().tolist(), index = df_new[groupby_col]).stack()
        # Now make sure the groupby col appears next to it 
        df_new = df_new.reset_index(groupby_col).reset_index(drop=True)
        df_new.columns=[groupby_col, mol_col]

    if split_stack_group: 
        # Split all the groups of interest 
        df_new = pd.DataFrame(df_new[groupby_col].str.split().tolist(), 
                              index = df_new[mol_col]).stack()
        # Now make sure the groupby col appears next to it 
        df_new = df_new.reset_index(mol_col).reset_index(drop=True)
        df_new.columns=[mol_col, groupby_col]

    # Now count the unique items! 
    unique_per_group = df_new.groupby(groupby_col).apply( lambda x : len(x.drop_duplicates(subset=mol_col)))
    ax = sns.distplot(unique_per_group.values, kde = False, norm_hist=False, 
                      rug=True)
    return ax

def plot_mol_weight(df : pd.DataFrame, col : str, split_stack_mol : bool =True):
    """Make a histogram of the molecular weight for all unique smiles in df[col]"""

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context="paper", font_scale=2, rc = {"figure.figsize" : (20,10)})
    df_new = df.copy()
    smiles_to_calc = set()
    for entry in df[col]: 
        if split_stack_mol: 
            subentries = entry.split()
        else: 
            subentries = [entry]
        smiles_to_calc.update(subentries)
    num_atoms = lambda x : Chem.MolFromSmiles(x).GetNumAtoms(onlyExplicit=0)

    # Num atoms: 
    vals_to_plot = []
    for mol in smiles_to_calc: 
        try: 
            new_weight = num_atoms(mol)  
            vals_to_plot.append(new_weight)
        except AttributeError: 
            # If the molecule is not able to be made to smiles
            pass

    ax = sns.distplot(vals_to_plot, kde = False, norm_hist=False,
                      rug=True)
    return ax 
            
def make_plots(rxn_df: pd.DataFrame, compounds : pd.DataFrame, out_prefix : str): 
    """ Make plots about the statistics for the col of interest
    Desired plots: 
    1. [DONE] Hist of # of unique inhibitors per ec class. 
    2. [DONE] Hist of # of unique substrates per ec class 
    3. [DONE] Hist of # of unique rxns each substrate participates in 
    4. [DONE] Hist of # of times sub/prod appears together  
    5. [DONE] Hist of mol weight of different compound types (inhibitors, subs, prods,
    etc.) 
    6. [DONE] Hist of # of genes per ec class  
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context="paper", font_scale=2, rc = {"figure.figsize" : (20,10)})

    # Length of genes
    fig = plt.figure()
    gene_set = set([i for i in rxn_df["SEQ_ID"] if type(i) is str and len(i) > 1] )
    gene_set_lens = [len(i) for i in gene_set]
    sns.distplot(gene_set_lens, rug=True)
    plt.xlabel(f"Gene lengths (# Amino Acids)")
    plt.ylabel(f"Freqs")
    fig.savefig(f"{out_prefix}_gene_sizes.png", 
                bbox_inches="tight")

    # Unique substrates
   
    fig = plt.figure()
    ax = plot_unique_elements(rxn_df, mol_col="SUBSTRATES",
                              groupby_col="EC_NUM", split_stack_group=False)
    ax.set_xlabel(f"Number of unique substrates per EC number") 
    ax.set_ylabel(f"Counts")
    fig.savefig(f"{out_prefix}_hist_unique_substrates.png",
                bbox_inches="tight")

    # Unique genes per EC 
    fig = plt.figure()
    ax = plot_unique_elements(rxn_df, mol_col="SEQ_ID",
                              groupby_col="EC_NUM", split_stack_group=False)
    ax.set_xlabel(f"Number of unique genes per EC number") 
    ax.set_ylabel(f"Counts")
    fig.savefig(f"{out_prefix}_hist_unique_genes.png",
                bbox_inches="tight")

    # Unique inhibitors
    inhibitors_df = compounds[compounds["COMPOUND_TYPE"] == "INHIBITORS"]
    inhibitors_df = inhibitors_df.reset_index(drop=True) 

    fig = plt.figure()
    ax = plot_unique_elements(inhibitors_df, mol_col="COMPOUND",
                              groupby_col="EC_NUM", split_stack_group=False)
    ax.set_xlabel(f"Number of unique inhibitors per EC number") 
    ax.set_ylabel(f"Counts")
    fig.savefig(f"{out_prefix}_hist_unique_inhibitors.png", bbox_inches="tight")

    # Now do # of unique reactions for substrates
    rxn_df["RXNS"] = rxn_df["SUBSTRATES"].str.strip() + " = " + rxn_df["PRODUCTS"].str.strip()

    # Here we want to grup by substrates (splitting that column) and count the
    # of unique reactions (not split)
    fig = plt.figure()
    ax = plot_unique_elements(rxn_df, mol_col="RXNS", groupby_col="SUBSTRATES", 
                              split_stack_mol=False, split_stack_group=True)
    ax.set_xlabel(f"Number of unique reactions per substrate") 
    ax.set_ylabel(f"Counts")
    fig.savefig(f"{out_prefix}_hist_unique_rxns_per_sub.png", bbox_inches="tight")
    fig = plt.figure()

    # Plot times things are paired in a distribution
    rxn_pairings = get_rxn_pairs(
        rxn_df["RXNS"].dropna().drop_duplicates().values
    )
    # Number of times they co appear extracted
    fig = plt.figure()
    ax = sns.distplot([i[2] for i in rxn_pairings], kde=False, norm_hist=False, 
                      rug=True)
    ax.set_xlabel(f"Number of times a substrate and product co-occur") 
    ax.set_ylabel(f"Counts")
    fig.savefig(f"{out_prefix}_hist_co_occurences.png", bbox_inches="tight")

    # Get mol weight for different compounds
    fig = plt.figure()
    ax = plot_mol_weight(rxn_df, "SUBSTRATES", split_stack_mol=True)
    ax.set_xlabel(f"Number of atoms in substrate")
    ax.set_ylabel(f"Counts")
    fig.savefig(f"{out_prefix}_hist_num_atoms_sub.png", bbox_inches="tight")

    # Get mol weight for different compounds
    fig = plt.figure()
    ax = plot_mol_weight(inhibitors_df, "COMPOUND", split_stack_mol=True)
    ax.set_xlabel(f"Number of atoms in inhibitors")
    ax.set_ylabel(f"Counts")
    fig.savefig(f"{out_prefix}_hist_num_atoms_inhib.png", bbox_inches="tight")

def get_rxn_pairs(uniq_rxns : list) -> list:  
    """get_rxn_pairs.

    From a list of unique reactions, return a list that has
    (sub_item, prod_item, num_times_appear_together, max(num rxns with this sub, num for prod))
    We sort by the fraction of times they coappear 
    This is meant to be used to filter out pairings like O>>O=O

    Args:
        uniq_rxns (list): uniq_rxns

    Returns:
        list:
    """
    pairs_list = []

    # Count of times each sub/prod appears
    num_sub_rxns = defaultdict(lambda : 0)
    num_prod_rxns = defaultdict(lambda : 0)
    for rxn in uniq_rxns: 
        sub_str, prod_str = rxn.split(" = ")
        subs = sub_str.split()
        prods = prod_str.split()
        pairs_list.extend([(sub, prod) for sub in subs 
                           for prod in prods if sub and prod and sub != "UNK" and prod != "UNK"])

        # Increment count for all these
        for sub in subs: num_sub_rxns[sub] +=1 
        for prod in prods: num_prod_rxns[prod] +=1 

    mol_counts = Counter(pairs_list)

    # sub, prod, number, max(num rxns for sub, num rxns for prod)
    mol_counts = [(sub, prod, j, max(num_sub_rxns[sub], num_prod_rxns[prod])) 
                  for (sub,prod),j  in mol_counts.items()]

    # Sort by the fraction of times they coappear! 
    sorted_mol_counts = sorted(mol_counts, key= lambda x : x[2] / x[3])
    sorted_mol_counts = sorted_mol_counts[::-1]

    return sorted_mol_counts

def compare_brenda_stats(name_to_smiles_1 : str, 
                         name_to_smiles_2 : str, 
                         outfile : str, 
                         file_1_name : str = None, 
                         file_2_name : str = None) -> None:
    """compare_brenda_stats.

    Helper function (called from terminal) to compare two mappings from
    compound name to smiles

    Args:
        name_to_smiles_1 (str): name_to_smiles_1
        name_to_smiles_2 (str): name_to_smiles_2
        outfile (str): outfile
        file_1_name (str): file_1_name
        file_2_name (str): file_2_name

    Returns:
        None:
    """

    mapping_1 = utils.load_json(name_to_smiles_1) 
    mapping_2 = utils.load_json(name_to_smiles_2)
    file_1_name = file_1_name if file_1_name else name_to_smiles_1
    file_2_name = file_2_name if  file_2_name else name_to_smiles_2

    # In 1 and not in 2
    one_keys = set(mapping_1.keys())
    two_keys = set(mapping_2.keys())

    only_one = one_keys.difference(two_keys)
    only_two = two_keys.difference(one_keys)

    difference_keys = set()
    for key in one_keys.intersection(two_keys):
        if mapping_1[key] != mapping_2[key]: 
            difference_keys.add(key)

    # Now write to file
    with open(outfile, "w") as fp: 
        fp.write(f"Key, smiles only in file 1: {file_1_name}\n")
        only_1_entries = "\n".join([f"{k}:{mapping_1[k]}" for k in only_one])
        fp.write(only_1_entries)
        fp.write(f"\n\nKey, smiles only in file 2: {file_2_name}\n")
        only_2_entries = "\n".join([f"{k}:{mapping_2[k]}" for k in only_two])
        fp.write(only_2_entries)
        fp.write(f"\n\nKey, smiles that differ b/w {file_1_name} and {file_2_name}\n")
        difference_entries = "\n".join(
            [f"{file_1_name}-- {k}:{mapping_1[k]}\n {file_2_name}-- {k}:{mapping_2[k]}\n" 
             for k in difference_keys]
        )
        fp.write(difference_entries)

