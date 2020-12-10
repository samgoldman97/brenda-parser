""" Run analysis of number of reaction classes per

Currently:
- Preprocess data by removing all unknowns 
- Counting number of unique reactions and the fraction of the database that
  satisfies these


Usage: 
    python analyze_num_rxns --parsed-data results/out_rxn_final.tsv

"""

import argparse
import pandas as pd
import numpy as np

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--parsed-data", action="store", 
                        default="results/out_rxn_final.tsv", 
                        help="Name of reaction output file from parse")

    return parser.parse_args()

def count_reactions(data : pd.DataFrame): 
    """ count_reactions.

    Args:
        data (pd.DataFrame): Name of pandas data frame 
    """
    rxns_per_ec = dict()
    for ec_num, ec_group  in data.groupby("EC_NUM"):

        # Already counted is a set of tuples for substrates and reactants
        already_counted = set()
        unique_rxns = 0

        for index, entry in ec_group.iterrows(): 
            entry_subs = entry["SUBSTRATES"]
            entry_reactants = entry["PRODUCTS"]

            # If we haven't counted this, count it
            if (entry_subs, entry_reactants) not in already_counted:
                unique_rxns += 1

            # Add both directions to avoid the double count
            already_counted.add((entry_subs, entry_reactants))
            already_counted.add((entry_reactants, entry_subs))

        rxns_per_ec[ec_num] = unique_rxns

    # Count distribution
    counts = np.array(list(rxns_per_ec.values()))
    x = np.bincount(counts)
    upper_bound = len(x)

    # Ranges to compute
    check_names = ["1", "2-5", "6-10", ">10"]
    check_ranges = [(1,1), (2,5), (6,10), (11, upper_bound)]

    for check_name,  (check_start, check_end) in zip(check_names,
                                                     check_ranges): 
        out_val = x[check_start: check_end +1].sum() / x.sum()
        print(f"{check_name} Reactions: {out_val}")

def preprocess(data : pd.DataFrame) -> pd.DataFrame: 
    """ preprocess.

    Helper function to preprocess the data frame to remove duplicates 

    """
    # First, remove anything with unk 
    print(f"Len of data pre filter: {len(data)}")

    # First select rows where we don't have unk in reactants or products
    no_unk_prod = np.array([isinstance(i, str) and "UNK" not in i for i in data["PRODUCTS"]])
    no_unk_react = np.array([isinstance(i, str) and "UNK" not in i for i in data["SUBSTRATES"]])
    no_unk = np.logical_and(no_unk_prod, no_unk_prod)
    data = data[no_unk].reset_index(drop=True)

    # Now add a new column that has the reaction text
    # We need to sort the products and reactants
    sort_compounds = lambda x: ".".join(sorted(x.split(".")))
    reactions = [f"{sort_compounds(subs)}>>{sort_compounds(prods)}" 
                 for subs, prods in data[["SUBSTRATES", "PRODUCTS"]].values]

    data["reactions"] = reactions

    print(f"Len of data after pruning unknowns: {len(data)}")

    # Filter this down so that it's unique to the key of (EC_CLASS, reaction) 
    # I.e., remove duplicate reactions in each ec class
    # Only take the first one of each group arbitrarily
    data_grouped = (data 
                    .groupby(["EC_NUM", "reactions"])
                    .aggregate(lambda x : list(x)[0]))
    data = data_grouped.reset_index() 

    print(f"Len of data after removing duplicate rxns: {len(data)}")
    return data


def main(args : argparse.Namespace): 
    """ Complete analysis"""

    data = pd.read_csv(args.parsed_data, delimiter="\t", 
                       index_col=0)

    # Preprocess the data
    data = preprocess(data)
    
    # Run analysis
    count_reactions(data)

if __name__=="__main__": 
    args = get_args()
    main(args)

