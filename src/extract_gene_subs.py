# The MIT License (MIT)
# Copyright (c), 2020 Sam Goldman
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Convert the parsed brenda CSVs into gene substrate data. 

After parsing BRENDA, we may want to convert the output into data for
substrate-enzyme model formalism. To do this, we should remove co-factors

Example call signature: 
    python enzpred/data/brenda/extract_gene_subs.py --out-prefix data/processed/BRENDA/parsed_no_ligand/filtered --rxn-tsv-file data/processed/BRENDA/parsed_no_ligand/parsed_brenda_brenda_rxn_final.tsv --compound-tsv-file data/processed/BRENDA/parsed_no_ligand/parsed_brenda_brenda_rxn_compounds.tsv --smiles-to-names data/processed/BRENDA/parsed_no_ligand/parsed_brenda_smiles_to_comp_names.json

"""

import argparse
import pandas as pd
import numpy as np 
from collections import Counter, defaultdict
from typing import List, Tuple
import re
from rdkit import Chem
from rdkit.Chem import rdqueries
import molvs

from enzpred.utils import file_utils, model_utils, rxn_mapping
from enzpred.data.brenda import parse_brenda_stats, parse_brenda_genes 

# These are pairs that should not be filtered from iterative filtering and were
# filtered previously 
EXCLUDE_FILTER = [("2-ketoglutaric acid", "succinate"), 
                  ("reduced benzylviologen", "benzyl viologen oxidized"), 
                  ("reduced methyl viologen", "paraquat"), 
                  ("benzyl viologen oxidized", "reduced benzylviologen"), 
                  ("ubiquinone", "reduced ubiquinone"), 
                  ("thymidine", "thymine"), 
                  ("benzoquinone", "hydroquinone"), 
                  ("oxidized phenazine methosulphate", "methylsulfate"), 
                  ("phenazinemethosulfate", "reduced phenazine methosulfate"),
                  ("vitamin K3", "reduced menaphthone"), 
                  ("oxidized methyl viologen", "reduced methyl viologen"), 
                  ("pyrroloquinoline-quinone", "reduced pyrroloquinoline quinone"), 
                  ("succinic acid", "fumaric acid")]

# Remove these
MANUAL_REMOVAL = [("H2O2","H2O"), 
                  ("H2O2", "H2O2"), 
                  ("CoA", "acetyl-CoA"),
                  ("acetyl-CoA", "CoA"), 
                  ("GTP", "GDP"), 
                  ("dGTP", "dGDP"),
                  ("dATP", "dAMP"), 
                  ("AMP", "ATP"), 
                  ("ATP", "ADP"), 
                  ("ADP", "ATP"), 
                  ("ADP", "AMP"), 
                  ("ATP", "AMP"), 
                  ("UTP", "UMP"), 
                  ("UTP", "UDP"), 
                  ("IDP", "ITP"), 
                  ("CTP", "CMP"), 
                  ("CTP", "CMP"), 
                  ("NADH", "NADH")]

ATOM_CUTOFF_LOWER = 3
ATOM_CUTOFF_UPPER = 400
ATOM_CUTOFF_LOWER_KINETIC= 3
ATOM_CUTOFF_UPPER_KINETIC = 150

def get_args(): 
    """Get arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rxn-tsv-file", action="store", 
                       help="Parsed BRENDA tsv file")
    parser.add_argument("--compound-tsv-file", action="store", 
                       help="Parsed BRENDA tsv file")
    parser.add_argument("--smiles-to-names", action="store", 
                       help="json containing smiles to compoudn names", 
                        required=True)
    parser.add_argument("--count-method", action="store", 
                        help="counting method to use", 
                        default="NEIGHBORS")
    parser.add_argument("--out-prefix", action="store", 
                       help="Out prefix")
    parser.add_argument("--uniprot-file", action="store", 
                       help="Name of uniprot file containing active sites")
    parser.add_argument("--kinetic-data-file", action="store", 
                       help="Kinetic data input file")
    parser.add_argument("--debug", action="store_true", 
                        default=False, 
                       help="Debug mode")
    args = parser.parse_args()
    return args 

def explicit_removal(df : pd.DataFrame, 
                     name_to_smiles :dict, 
                     ) -> pd.DataFrame: 
    """explicit_removal.

    Filter from every rxn the explicit pairs detailed  

    Args:
        df (pd.DataFrame): df
        name_to_smiles (dict) : name_to_smiles

    Returns:
        pd.DataFrame: new df
    """

    # Items to remove
    remove_items = []
    for remove_sub, remove_prod in MANUAL_REMOVAL: 
        remove_sub_smiles = name_to_smiles[remove_sub]
        remove_prod_smiles = name_to_smiles[remove_prod]
        remove_items.append((remove_sub_smiles, remove_prod_smiles))

    subs = df["SUBSTRATES"].values
    prods = df["PRODUCTS"].values

    # Now filter from df
    new_subs, new_prods = [], []
    for  index, (sub, prod) in enumerate(zip(subs,prods)): 
        filtered_subs = set(sub.split())
        filtered_prods = set(prod.split())

        # remove all pairs we're filtering manually
        for remove_sub, remove_prod in remove_items: 
            ### If we have no products, remove these as if they had been there
            if remove_sub in filtered_subs and (remove_prod in filtered_prods or len(filtered_prods) <= 1):
                filtered_subs.discard(remove_sub)
                filtered_prods.discard(remove_prod)

        # If after filtering putative coenzyme rxn's, then use original 
        if len(filtered_subs) < 1: 
            filtered_subs = set(sub.split())
            filtered_prods = set(prod.split())

        new_subs.append(" ".join(filtered_subs))
        new_prods.append(" ".join(filtered_prods))

    df = df.copy()
    df["SUBSTRATES"] = new_subs
    df["PRODUCTS"] = new_prods

    return df

def iterative_filtering(df : pd.DataFrame, 
                        smiles_mapping :dict, 
                        rxn_num_cutoff : int = 50, 
                        pair_pct_cutoff: float = 0.8
                        ) -> Tuple[pd.DataFrame, list]:
    """iterative_filtering.

    Iteratively filter out the most observed pairing of reactant to product

    Args:
        df (pd.DataFrame): df
        smiles_mapping (pd.DataFrame): smiles_mapping
        rxn_num_cutoff (int): # of reaction apperances needed for sub or prod needed to prune 
        pair_pct_cutoff (float): % of coappearances needed to prune

    Returns:
        Tuple[pd.DataFrame, list]: new df, filtered out rxns
    """
    subs = df["SUBSTRATES"]
    prods = df["PRODUCTS"]

    pd_rxns = (subs + " = " +  prods)
    uniq_rxns = pd.unique(pd_rxns)
    uniq_rxns = [i for i in uniq_rxns if type(i) is str] 

    # Fn to check if a substrate name list and a product name list should be
    # excluded. Return true if they should be excluded
    in_exclusion = lambda sub_list, prod_list: np.any(
        [prot_sub in sub_list and prot_prod in prod_list for prot_sub, prot_prod in EXCLUDE_FILTER]
    )

    filtered_out_pairs = []
    while True: 
        rxn_counts = parse_brenda_stats.get_rxn_pairs(uniq_rxns)
        # Refilter for > rxn_num_cutoff rxns 
        rxn_counts = [tmp_rxn for tmp_rxn in rxn_counts 
                      if tmp_rxn[3] > rxn_num_cutoff]
        # Refilter for % of times they pair together
        rxn_counts = [tmp_rxn for tmp_rxn in rxn_counts 
                      if tmp_rxn[2] / tmp_rxn[3] > pair_pct_cutoff]

        # Refilter out protected pairs
        rxn_counts = [tmp_rxn for tmp_rxn in rxn_counts 
                    if not in_exclusion(smiles_mapping.get(tmp_rxn[0], []), smiles_mapping.get(tmp_rxn[1], []))]

        # If we have no more, exit
        if not rxn_counts:
            break

        labeled_tuple_counts = [
            (smiles_mapping.get(sub_name, ""), smiles_mapping.get(prod_name, ""),
             sub_name, prod_name, cnt, total_cnt) 
            for sub_name, prod_name, cnt,total_cnt in rxn_counts]

        # Filter out top reaction pair from all reactions
        filter_sub, filter_prod = rxn_counts[0][0], rxn_counts[0][1]
        filtered_out_pairs.append(labeled_tuple_counts[0])
        new_pd_rxns = []

        # Filter out of pd 
        for rxn in pd_rxns: 
            if type(rxn) is str: 
                subs, prods = rxn.split(" = ")

                sub_list = set(subs.split())
                prod_list = set(prods.split())
                # Remove if we find both in subs and products OR we find in 
                # subs and no prods listed
                if filter_sub in sub_list and (filter_prod in prod_list or len(prod_list) == 0) :
                    sub_list =  [j for j in sub_list if j != filter_sub]
                    prod_list = [j for j in prod_list if j != filter_prod]
                new_subs = " ".join(sub_list)
                new_prods = " ".join(prod_list)

                new_pd_rxns.append(" = ".join([new_subs, new_prods]))
            else: 
                # Shouldn't get here anymore
                raise ValueError
        uniq_rxns = list(set(new_pd_rxns))
        pd_rxns = new_pd_rxns

    # Now reset the return df 
    new_subs, new_prods = [], []
    for j in pd_rxns: 
        new_sub, new_prod = j.split(" = ")
        new_subs.append(new_sub)
        new_prods.append(new_prod)

    df = df.copy()
    df["SUBSTRATES"] = new_subs
    df["PRODUCTS"] = new_prods

    return df, filtered_out_pairs

def remove_empty(df: pd.DataFrame, column : str) -> pd.DataFrame: 
    """ Remove all els in the column that are empty or don't have type str""" 

    col_data = df[column]
    non_unk = [type(i) is str and len(i.strip()) > 0 for i in col_data]

    df = df[non_unk].reset_index(drop=True)
    return df 

def remove_transport_rxns(df : pd.DataFrame) -> pd.DataFrame: 
    """ Remove all transport rxns """
    non_transport =  rxn_data["EC_NUM"].apply(
        lambda x : x[0] != "7")
    return rxn_data[non_transport].reset_index(drop=True)

def remove_cofactors(df : pd.DataFrame, cofactor_df : pd.DataFrame) -> pd.DataFrame: 
    """ Remove all cofactors"""

    ###### Get rid of all cofactors ##### 
    # Make mapping
    org_ec_cofactor = defaultdict(lambda : [])
    for row_num, row in cofactor_df.iterrows(): 
        org_ec_cofactor[row["ORG_EC_KEY"]].append(row["COMPOUND"])

    rxn_data = df
    # Filter these out from all rows
    new_subs,new_prods = [], []
    for index, rxn in rxn_data.iterrows(): 
        subs = rxn_data.loc[index, "SUBSTRATES"]
        prods = rxn_data.loc[index, "PRODUCTS"]
        org_ec = rxn_data.loc[index, "ORG_EC_KEY"]
        for cofactor in org_ec_cofactor.get(org_ec, []): 
            re_cofactor = re.escape(cofactor)
            subs = re.sub(rf"(^|\s){re_cofactor}($|\s)", " ", subs).strip()
            prods = re.sub(rf"(^|\s){re_cofactor}($|\s)", " ", prods).strip()

        new_subs.append(subs)
        new_prods.append(prods)

    rxn_data = df.copy()
    rxn_data["SUBSTRATES"] = new_subs
    rxn_data["PRODUCTS"] = new_prods
    return rxn_data


# Fn to get the number of atoms
num_atoms_heavy = lambda mol: mol.GetNumAtoms(onlyExplicit=1)
num_atoms_all = lambda mol: mol.GetNumAtoms(onlyExplicit=0)

def remove_size_comps(rxn_data : pd.DataFrame, atom_cutoff_lower: int, 
                      atom_cutoff_upper : int, column: str ="SUBSTRATES", 
                      heavy_only : bool = False) -> pd.DataFrame: 
    """ Remove all substrate compounds with <= atom_cutoff_lower atoms or
    invalid smiles or >= atom_cutoff_upper --> Do not remove UNK"""

    unique_subs = set([i for sub in rxn_data[column] 
                         for i in sub.split()])

    # Remove based on cutoffs
    to_remove = []
    for j in unique_subs:
        mol = Chem.MolFromSmiles(j)
        if heavy_only: 
            num_atoms_mol = None if not mol else num_atoms_heavy(mol)
        else:
            num_atoms_mol = None if not mol else num_atoms_all(mol)
        if (j != "UNK" and 
                (not num_atoms_mol or 
                num_atoms_mol <= atom_cutoff_lower or 
                num_atoms_mol >= atom_cutoff_upper)
                ):
            to_remove.append(j)
    to_remove = set(to_remove)

    # This has gotten time inefficient??
    new_subs = []
    for index, row in rxn_data.iterrows():
        subs = row[column]
        sub_list = set(subs.split())
        sub_list.difference_update(to_remove)
        new_subs.append(" ".join(sub_list))
        # for mol in to_remove: 
        #     mol_escape= re.escape(mol)
        #     subs = re.sub(rf"(^|\s){mol_escape}($|\s)", " ", subs).strip()
        # new_subs.append(subs)

    rxn_data = rxn_data.copy()
    rxn_data[column] = new_subs
    return rxn_data

def get_compounds_with_genes(rxn_data :pd.DataFrame, 
                             compounds : pd.DataFrame) -> pd.DataFrame:
    """Subset the compounds AR to only have entries that do have a gene"""
    compounds = model_utils.subset_df_overlap(compounds, rxn_data, 
                                              "ORG_EC_KEY", "ORG_EC_KEY")

    # valid_ids = rxn_data["ORG_EC_KEY"].values
    # valid_compounds = [id_ in valid_ids for id_ in compounds["ORG_EC_KEY"]]

    # # Extract only the valid compounds
    # compounds = compounds[valid_compounds].reset_index(drop=True)
    return compounds

def size_cutoff_genes(df : pd.DataFrame, seq_col : str, lower_bound: int = 50,
                      upper_bound : int = 1000) -> pd.DataFrame:
    """Subset the df such that all sequences with < a certain length are set to
    be empty """
    df_col = df[seq_col].values
    new_list = []
    for item in df_col: 
        if len(item) < lower_bound or len(item) > upper_bound: 
            new_list.append("")
        else: 
            new_list.append(item)

    df_new = df.copy()
    df_new[seq_col] = new_list
    return df_new 

def count_subs(col : pd.Series, 
                    method : str="COUNT") -> List[Tuple[str, int]] : 
    """count_subs.

    This is an old method to get counts of how many times the substrate appears
    in the column or how many neighbors (connected substrates) each substrate
    had
    
    Args:
        col (pd.Series): col
        method (str): method can be in ["COUNT", "NEIGHBORS", etc) 

    Returns:
        List[Tuple[str, int]]:
    """

    if method == "COUNT": 
        # Remove enzyme redundancy by only getting uniq substrates
        uniq_subs = pd.unique(col)
        # Total list of all compounds used 
        comp_list = [comp for sub_str in uniq_subs for comp in sub_str.split()]

        # Return the counts of each item 
        mol_counts = Counter(comp_list)

    elif method == "NEIGHBORS": 
        # Total list of all compounds used 
        neighbor_set_dict = defaultdict(lambda : set())
        for sub_str in col.values: 
            comps = sub_str.split()
            for comp in comps: 
                neighbor_set_dict[comp].update(comps)
        # Now get counts
        mol_counts = {k : len(v) for k,v in neighbor_set_dict.items()}

    else: 
        raise NotImplemented 

    # From highest counts to lowest counts 
    sorted_mol_counts = sorted([(i,j) for i,j  in mol_counts.items()], 
                               key= lambda x : x[1])[::-1]
    return sorted_mol_counts

def filtered_to_str(filtered : list) -> str: 
    def stringify_entry(x : Tuple) ->str: 
        substrate_names = ", ".join(x[0])
        product_names = ", ".join(x[1])
        substrate_smiles = str(x[2])
        product_smiles = str(x[3])
        co_occurence = str(x[4])
        max_occurences = str(x[5])
        fstring = (f"Substrate Synonyms: {substrate_names}\n"
                   f"Product Synonyms: {product_names}\n"
                   f"Substrate Smiles: {substrate_smiles}\n"
                   f"Product Smiles: {product_smiles}\n"
                   f"Co-occurence number: {co_occurence}\n"
                   f"Max(product occurences, substrate occurences): {max_occurences}\n")

        return fstring

    return "\n\n".join([stringify_entry(i) for i in filtered])

def get_multi_substrate_list(rxn_data : pd.DataFrame, 
                             smiles_mapping : dict) -> pd.DataFrame: 
    """get_multi_substrate_list.

    Args:
        rxn_data (pd.DataFrame): rxn_data
        smiles_mapping (dict): smiles_mapping

    Returns:
        pd.DataFrame:
    """
    # Export all reactions with 2 substrates
    output_rows = []
    for row, entry in rxn_data.iterrows():
        substrates = entry["SUBSTRATES"].split() 
        if len(substrates) > 2: 
            sub_names = " + ".join([smiles_mapping[i][0] for i in substrates] )
            save_dict = {'RXN_TEXT' : entry["RXN_TEXT"],
                         "substrates" : entry["SUBSTRATES"], 
                         "sub_names" : sub_names, 
                         "ec class": entry["EC_NUM"]
                         }
            output_rows.append(save_dict)
    return pd.DataFrame(output_rows)


def filter_catalyzed_inhibs(rxn_data : pd.DataFrame, 
                            compounds : pd.DataFrame) -> pd.DataFrame : 
    """filter_catalyzed_inhibs.

    Remove all inhibitors that are also catalyzed by the same ec class

    Args:
        rxn_data (pd.DataFrame): rxn_data
        compounds (pd.DataFrame): compounds

    Returns:
        pd.DataFrame:
    """

    # Enz to sub, Enz to Inhib
    enz_to_sub = defaultdict(lambda : set())
    enz_to_inhib = defaultdict(lambda : set())
    ec_to_sub = defaultdict(lambda : set())
    ec_to_inhib = defaultdict(lambda : set())

    # EC NUM TO SUB
    for subs, enzyme, ec  in zip(rxn_data["SUBSTRATES"].values,
                                 rxn_data["SEQ"].values, 
                                 rxn_data["EC_NUM"].values):         
        for sub in subs.split(): 
            enz_to_sub[enzyme].add(sub)
            ec_to_sub[ec].add(sub)

    for inhib, enzyme, comp_type, ec in zip(compounds["COMPOUND"].values, 
                                            compounds["SEQ"].values, 
                                            compounds["COMPOUND_TYPE"].values, 
                                            compounds["EC_NUM"]): 
        if comp_type == "INHIBITORS": 
            enz_to_inhib[enzyme].add(inhib)
            ec_to_inhib[ec].add(inhib)

    # Create removal set --> set([(ec, sub_smile)])
    removal_set = set() 
    for ec in ec_to_inhib:
        intersect = ec_to_inhib[ec].intersection(ec_to_sub.get(ec, set()))
        print(f"Size of intersection b/w inhibs and subs: {len(intersect)} / {len(ec_to_inhib[ec])}")
        for i in intersect: 
            removal_set.add((ec, i))

    print(f"Number of inhibitors to remove: {len(removal_set)}")

    # Now remove all inhibitors that also act as substrates
    kept_subs = []
    for index, (comp, comp_type, ec_num) in enumerate(zip(compounds["COMPOUND"].values, 
                                                          compounds["COMPOUND_TYPE"].values, 
                                                          compounds["EC_NUM"].values)): 
        if comp_type  == "INHIBITORS": 
            if (ec_num, comp) not in removal_set: 
                kept_subs.append(index)
        else: 
            # Always keep if not inhib
            kept_subs.append(index)

    print(f"Number of kept indices in inhibitors:  {len(kept_subs)} / {len(compounds)}")
    compounds = compounds.loc[kept_subs].reset_index(drop=True)
    return compounds


def main(args): 
    """Main method"""
    # Make out directory
    file_utils.make_dir(args.out_prefix)
    rxn_data = pd.read_csv(args.rxn_tsv_file,
                           index_col=0, 
                           sep="\t")
    compounds = pd.read_csv(args.compound_tsv_file, 
                            index_col=0, 
                            sep="\t")

    # Fill na with string
    rxn_data.fillna(value="", inplace=True)
    compounds.fillna(value="", inplace=True)

    # Debugging
    if args.debug: 
        rxn_data = rxn_data.sample(frac=0.001, replace=False).reset_index(drop=True)
        compounds = compounds.sample(frac=0.001, replace=False).reset_index(drop=True)


    compounds = size_cutoff_genes(compounds, "SEQ", lower_bound = 50,
                                  upper_bound=1000)

    smiles_mapping = file_utils.load_json(args.smiles_to_names) 
    name_to_smiles = {v_sub : k for k,v in smiles_mapping.items() for v_sub in v}

    # Now get cofactors 
    cofactors = compounds[compounds["COMPOUND_TYPE"] == "COFACTOR"]

    # Pre removal of unknown
    parse_brenda_stats.make_plots(rxn_data,compounds,  
                                  f"{args.out_prefix}_PREFILTER")

    rxn_data = size_cutoff_genes(rxn_data, "SEQ", lower_bound = 50,
                                 upper_bound=1000)
    compounds = size_cutoff_genes(compounds, "SEQ", lower_bound = 50,
                                  upper_bound=1000)

    # Remove inhibitors that are also substrates for the same ec class
    compounds = filter_catalyzed_inhibs(rxn_data, compounds) 

    # Get rid of all rxn's that have UNK in them 
    # Remove all entries that have unk in substrate    
    # Products can still have unk na
    rxn_data = model_utils.remove_unk_nan(rxn_data, col="SUBSTRATES")

    # Try iterative filtering to knock out the H2O to O2 type mappings
    rxn_data, filtered = iterative_filtering(rxn_data, smiles_mapping, 
                                             rxn_num_cutoff=50, pair_pct_cutoff=0.5)

    # Report all filtered compounds from iterative filtering
    with open(f"{args.out_prefix}_iteratively_filtered.txt", "w") as fp: 
        fp.write(filtered_to_str(filtered))

    # Remove pairs explicitly   
    rxn_data = explicit_removal(rxn_data, name_to_smiles)

    # Remove cofactors from both product and substrate where named
    rxn_data = remove_cofactors(rxn_data, cofactors)

    # Remove all mols <= 3
    rxn_data = remove_size_comps(rxn_data,atom_cutoff_lower=ATOM_CUTOFF_LOWER,
                                 atom_cutoff_upper=ATOM_CUTOFF_UPPER, column="SUBSTRATES") 
    rxn_data = remove_size_comps(rxn_data,atom_cutoff_lower=ATOM_CUTOFF_LOWER, 
                                 atom_cutoff_upper=ATOM_CUTOFF_UPPER, column="PRODUCTS") 

    # Remove inhibitors when inhibitors are also members of the same EC 
    #; Drop duplicates
    rxn_data = rxn_data.drop_duplicates(subset=["SUBSTRATES","PRODUCTS",
                                                 "EC_NUM", "SEQ"])
    rxn_data = rxn_data.reset_index(drop=True)
    # Reset index and remove empty ones..
    rxn_data = remove_empty(rxn_data, "SUBSTRATES")

    ### Handle compounds
    # Remove all mols <= 3
    compounds = remove_size_comps(compounds, atom_cutoff_lower=ATOM_CUTOFF_LOWER,
                                  atom_cutoff_upper=ATOM_CUTOFF_UPPER, column="COMPOUND") 

    compounds = remove_empty(compounds, "COMPOUND")
    compounds = compounds.drop_duplicates(subset=["COMPOUND","SEQ", "COMPOUND_TYPE"])
    compounds = compounds.reset_index(drop=True)
    # compounds = get_compounds_with_genes(rxn_data, compounds) 

    # Export all rxns that still have > 2 substrates 
    get_multi_substrate_list(rxn_data,
                             smiles_mapping).to_csv(f"{args.out_prefix}_multi_substrates.tsv", sep="\t")

    # Add rxn centers to rxn_data
    new_vals = rxn_mapping.add_atom_mappings(rxn_data, timeout=0.5)
    rxn_data["RXN_CENTERS"] = new_vals

    new_vals= parse_brenda_genes.get_active_site_list(rxn_data["SEQ_ID"].values,
                                            args.uniprot_file)
    rxn_data["ACTIVE_SITES"] = new_vals

    # Export full data
    rxn_data.to_csv(f"{args.out_prefix}_full_model_data.tsv", sep="\t")
    compounds.to_csv(f"{args.out_prefix}_full_compound_data.tsv", sep="\t")

    # POST FILTER -> First make plots
    parse_brenda_stats.make_plots(rxn_data,compounds,  
                                  f"{args.out_prefix}_POSTFILTER")

    # Read in directly
    rxn_data = pd.read_csv(f"{args.out_prefix}_full_model_data.tsv", sep="\t",
                           index_col=0)
    compounds = pd.read_csv(f"{args.out_prefix}_full_compound_data.tsv", sep="\t",
                           index_col=0)

    # Fill na with string
    rxn_data.fillna(value="", inplace=True)
    compounds.fillna(value="", inplace=True)

    ### Now compute stats
    # CS MODEL: 
    CS_rxn_data = model_utils.split_col(rxn_data, "SUBSTRATES")
    CS_rxn_data = CS_rxn_data.drop_duplicates(subset=["EC_NUM",
                                                      "SUBSTRATES"]).reset_index(drop=True)
    stats_output = parse_brenda_stats.get_rxn_compound_stats(CS_rxn_data, compounds)
    stats_output = {k.replace("rxn", "sub") : (float(v) if type(v) is not
                                               pd.Series else "nan") 
                    for k, v in stats_output.items()}
    file_utils.dump_json(stats_output, f"{args.out_prefix}_CS_stats.json")

    # ES MODEL: 
    ES_rxn_data = model_utils.remove_unk_nan(rxn_data, "SEQ")
    ES_rxn_data = model_utils.split_col(ES_rxn_data, "SUBSTRATES") 
    ES_rxn_data = ES_rxn_data.drop_duplicates(subset=["SEQ",
                                                      "SUBSTRATES"]).reset_index(drop=True)
    ES_compounds = get_compounds_with_genes(ES_rxn_data, compounds) 
    stats_output = parse_brenda_stats.get_rxn_compound_stats(ES_rxn_data, ES_compounds)
    stats_output = {k.replace("rxn", "sub") : (float(v) if type(v) is not
                                               pd.Series else "nan") 
                    for k, v in stats_output.items()}
    file_utils.dump_json(stats_output, f"{args.out_prefix}_ES_stats.json")

    # ESP MODEL: 
    ESP_rxn_data = model_utils.remove_unk_nan(rxn_data, "SEQ")
    ESP_rxn_data = model_utils.remove_unk_nan(ESP_rxn_data, "PRODUCTS")
    ESP_rxn_data = model_utils.split_col(ESP_rxn_data, "SUBSTRATES") 
    ESP_compounds = get_compounds_with_genes(ESP_rxn_data, compounds) 
    stats_output = parse_brenda_stats.get_rxn_compound_stats(ESP_rxn_data, ESP_compounds)
    stats_output = {k.replace("rxn", "sub") : (float(v) if type(v) is not
                                               pd.Series else "nan") 
                    for k, v in stats_output.items()}
    file_utils.dump_json(stats_output, f"{args.out_prefix}_ESP_stats.json")

    # Get counts for how many times each smiles appears in unique rows
    # Do this for debugging
    # tuple_counts = count_subs(rxn_data["SUBSTRATES"], args.count_method)
    # labeled_tuple_counts = [(smiles_mapping.get(name, []), name, cnt)
    #                         for name, cnt in tuple_counts]

def get_kinetic_df_cs(df): 
    """ Get the CS formulation df by aggregating over EC num"""
    groups = []
    for group_name, group_df in df.groupby("KINETIC_PARAM"): 

        if group_name == "KM_VALUE": 
            fn = lambda x: x.idxmin()
        else: 
            fn = lambda x: x.idxmax()

        new_df_idx = fn(group_df.groupby(["EC_NUM", "COMPOUND"])["KINETIC_VAL"])
        new_df = df.loc[new_df_idx].reset_index(drop=True)

        groups.append(new_df)
    cs_df = pd.concat(groups).reset_index(drop=True)
    return cs_df

def split_kinetic(df, cat_column="KINETIC_PARAM", val_column="KINETIC_VAL"): 
    """ Take cat_column and pivot it into new independent columns with no aggregation

    Helpful for the multi task problem

    """
    grouped_df = df.groupby(cat_column)
    for k, v in grouped_df.groups.items(): 
        # Get all the kinetic vals where the param is k 
        param_vals = df.loc[v][val_column].values
        # Now add a new column k to the original df
        df.loc[v, k] = param_vals

def get_kinetic_df_es(df): 
    """ Get the ES formulation df by aggregating over SEQ """

    # Get only places where seq exists!
    df = df.query("SEQ != ''").reset_index()
    groups = []
    for group_name, group_df in df.groupby("KINETIC_PARAM"): 
        if group_name == "KM_VALUE": 
            fn = lambda x: x.idxmin()
        else: 
            fn = lambda x: x.idxmax()

        new_df_idx = fn(group_df.groupby(["SEQ", "COMPOUND"])["KINETIC_VAL"])
        new_df = df.loc[new_df_idx].reset_index(drop=True)
        groups.append(new_df)
    es = pd.concat(groups).reset_index(drop=True)
    return es 

def remove_cofactors_kinetic(df : pd.DataFrame, cofactor_df : pd.DataFrame,
                             ec_level : int = 3) -> pd.DataFrame: 
    """ Remove all cofactors. Do this at the level of the 3rd ec class"""

    ec_level = 3
    extract_ec = lambda ec: ".".join(ec.split(".")[:ec_level])
    ###### Get rid of all cofactors ##### 
    # Make mapping
    ec_cofactor = defaultdict(lambda : set())
    for row_num, row in cofactor_df.iterrows(): 
        row_ec = extract_ec(row["EC_NUM"])
        row_compound = row["COMPOUND"]
        compound_name = row["COMPOUND_TEXT"]
        ec_cofactor[row_ec].add(row_compound)

    should_be_filtered = []
    for comp, ec in df[["COMPOUND", "EC_NUM"]].values: 
        ec_num = extract_ec(ec)
        should_be_filtered.append(comp in ec_cofactor[ec_num])
    rows_to_filter = np.array(should_be_filtered)

    return df[~rows_to_filter]

def no_carbon(df): 
    """ Filter all items with no carbon"""
    q = rdqueries.AtomNumEqualsQueryAtom(6)
    return df[np.array([len(Chem.MolFromSmiles(i).GetAtomsMatchingQuery(q)) > 0  for i in df["COMPOUND"]])]

def many_fragments(df, frag_thresh =3): 
    """ Filter smiles strings that have >thresh fragments (e.g. hexakis(benzylammonium) decavanadate (V) dihydrate in 1.4.3.21)"""
    return df[np.array([len(i.split(".")) <= frag_thresh for i in df["COMPOUND"]])]


def clean_kinetic_data(args): 
    """ Clean kinetic data 

    1. [DONE] Remove sequences that are clearly incomplete (e.g. <50 and >1000) 
    2. [DONE] Remove very small substrates and very large substrates 
    3. [DONE] Add in active sites
    4. [DONE] Produce statistics 
    5. [DONE] Produce plots

    TODO: 
    - Add in reaction centers taking into account that data? 
    - Remove NADPH and other cofactors 


    python enzpred/data/brenda/extract_gene_subs.py --kinetic-data-file data/interim/BRENDA/parsed_with_ligand_km/parsed_brenda_brenda_kinetic_df.tsv --smiles-to-names data/interim/BRENDA/parsed_with_ligand_km/parsed_brenda_smiles_to_comp_names.json --out-prefix data/processed/BRENDA/parsed_with_ligand_km/filtered --uniprot-file data/raw/Uniprot/brenda_uniprot_seqs.txt
    """

    file_utils.make_dir(args.out_prefix)

    kinetic_df = pd.read_csv(args.kinetic_data_file, sep="\t", index_col = 0 )

    # Remove cofactors
    compounds = pd.read_csv(args.compound_tsv_file, 
                            index_col=0, 
                            sep="\t")

    # Now get cofactors 
    cofactors = compounds.query("COMPOUND_TYPE == 'COFACTOR'")

    # Debugging
    if args.debug: 
        kinetic_df= kinetic_df.sample(frac=0.001, replace=False).reset_index(drop=True)
        cofactors = cofactors.sample(frac=0.001, replace=False).reset_index(drop=True)

    # Filter out cofactors
    kinetic_df = remove_cofactors_kinetic(kinetic_df,
                                          cofactors).reset_index(drop=True)

    # Filter out transporters
    non_transport = kinetic_df["EC_NUM"].str.split(".").apply(lambda x : int(x[0]) != 7)
    kinetic_df = kinetic_df[non_transport].reset_index()

    # Fill na with string
    kinetic_df.fillna(value="", inplace=True)

    # Split the df by expanding the KM/KCAT into independent columns
    split_kinetic(kinetic_df)

    # Export stats
    parse_brenda_stats.make_kinetic_plots(kinetic_df,  
                                          f"{args.out_prefix}_kinetic_PREFILTER")


    # Filter df 
    kinetic_df = size_cutoff_genes(kinetic_df, "SEQ", lower_bound = 50,
                                   upper_bound=1000)

    # Filter for negative kinetic parameters
    kinetic_df = kinetic_df.query("KINETIC_VAL >= 0").reset_index(drop=True)

    # Remove substrates that are very small and very large
    # Remove all mols <= 3
    # Don't use stringent upward filter
    kinetic_df = remove_size_comps(kinetic_df,atom_cutoff_lower=ATOM_CUTOFF_LOWER_KINETIC,
                                 atom_cutoff_upper=ATOM_CUTOFF_UPPER_KINETIC * 4,
                                 column="COMPOUND") 

    # Use HEAVY filter for the upper bound
    kinetic_df = remove_size_comps(kinetic_df,atom_cutoff_lower=0,
                                   atom_cutoff_upper=ATOM_CUTOFF_UPPER_KINETIC,
                                   column="COMPOUND", 
                                   heavy_only = True) 

    # Remove all the wacky fragments
    kinetic_df = many_fragments(kinetic_df).reset_index(drop=True)

    # Remove mols w/ no carbon
    kinetic_df = no_carbon(kinetic_df).reset_index(drop=True)

    # Remove empty entries
    kinetic_df = remove_empty(kinetic_df, column="COMPOUND")

    # Use uniprot file here to add back in 
    new_vals = parse_brenda_genes.get_active_site_list(kinetic_df["SEQ_ID"].values,
                                            args.uniprot_file)

    kinetic_df["ACTIVE_SITES"] = new_vals

    # High bar for duplicates
    kinetic_df= kinetic_df.drop_duplicates(subset=["COMPOUND","KINETIC_VAL",
                                                   "KINETIC_PARAM", "EC_NUM",
                                                   "SEQ"]).reset_index(drop=True)

    # Export full data
    kinetic_df.to_csv(f"{args.out_prefix}_full_model_kinetic_data.tsv", sep="\t")

    # Export stats
    parse_brenda_stats.make_kinetic_plots(kinetic_df,  
                                          f"{args.out_prefix}_kinetic_POSTFILTER")

    # Now read in directly
    kinetic_df = pd.read_csv(f"{args.out_prefix}_full_model_kinetic_data.tsv", sep="\t",
                             index_col=0)

    # Filter for CS 
    kinetic_df_cs =  get_kinetic_df_cs(kinetic_df)
    kinetic_stats = parse_brenda_stats.get_kinetic_stats(kinetic_df_cs)
    file_utils.dump_json(kinetic_stats, f"{args.out_prefix}_CS_kinetic_stats.json")

    # Filter for ES 
    kinetic_df_es =  get_kinetic_df_es(kinetic_df)
    kinetic_stats = parse_brenda_stats.get_kinetic_stats(kinetic_df_es)
    file_utils.dump_json(kinetic_stats, f"{args.out_prefix}_ES_kinetic_stats.json")
    return 


if __name__=="__main__": 
    args = get_args()
    clean_kinetic_data(args)
    #main(args)
    


