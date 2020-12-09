""" parse_brenda_main.py 

    Helper script to parse the brenda file

    Typical usage example:

    Call to run: 

    python parse_brenda.py --brenda-flat-file data/raw/brenda_download.txt --brenda-ligands data/raw/brenda_ligands.csv --out-prefix results/out --no-standardize --load-prev --opsin-loc external_tools/opsin.jar

    Sample call while debugging:
    
    python -m pdb parse_brenda.py --brenda-flat-file data/raw/brenda_download_short.txt --debug --brenda-lig data/raw/brenda_ligands.csv --out-prefix results/brenda_parsed --no-standardize --load-prev
"""

import os
import sys
import re
import argparse
import typing
from typing import Tuple, Dict, Optional, Callable, List
from collections import defaultdict
import logging
import time
import json
import pandas as pd
import numpy as np
from rdkit import Chem


from src import utils
from src import parse_brenda_flatfile
from src import parse_brenda_compounds
from src import parse_brenda_stats

REACTION_HEADERS = parse_brenda_flatfile.REACTION_HEADERS
COMPOUND_HEADERS = parse_brenda_flatfile.COMPOUND_HEADERS
KINETIC_HEADERS = parse_brenda_flatfile.KINETIC_HEADERS

def add_compounds(enzymes_data: dict, 
                  mapped_compounds: dict) -> Tuple[dict, dict, dict]:
    """add_genes_compounds.

    Return an updated version of the enzymes data with the unique and
    identified reactions outputted as well the activating and inhibiting
    compounds. 

    Args:
        enzymes_data (dict): enzymes_data
        mapped_compounds (dict): mapped_compounds

    Returns:
        Tuple[dict, dict, dict]: updated enzymes data, rxn_set,  and compound
            set. The rxn set and compound set are the final outputs of the
            model
    """

    # Extract smiles vzn of a list; if unknown, add "UNK"
    # NOTE: Replace " " in smiles strings with "" for paranoia reasons
    extract_smiles = lambda x: [
        mapped_compounds.get(i, "UNK").replace(" ", "") for i in x
    ]
    concat_comments = lambda x: ".".join(comment["COMMENT"] for comment in x)

    def is_equal_rxn(entry_1, entry_2):
        """ quick helper fn to see if substrate-product pair 1 == pair 2"""
        s1, s2 = set(entry_1["SUBSTRATES"]), set(entry_2["SUBSTRATES"])
        p1, p2 = set(entry_1["PRODUCTS"]), set(entry_2["PRODUCTS"])
        return (s1 == s2) and (p1 == p2)

    def is_equal_comp(entry_1, entry_2):
        """ quick helper fn to see if compound entry 1 == compound entry 2"""
        s1, s2 = set(entry_1["SMILES"]), set(entry_2["SMILES"])
        return s1 == s2

    def update_refs(entry_list, new_entry, is_equal) -> None:
        """ Scroll through cur entry list, compare entries. If eq, add to refs"""
        for old_entry in entry_list:
            if is_equal(old_entry, new_entry):
                old_entry["REFS"] = list(
                    set(old_entry["REFS"]).union(new_entry["REFS"]))
                # Because these are dicts, need for loop to only keep unique
                # comments
                for new_comment in new_entry["COMMENTS"]:
                    if new_comment not in old_entry["COMMENTS"]:
                        old_entry["COMMENTS"].append(new_comment)
                return None

        entry_list.append(new_entry)
        return None

    rxn_set = []
    compound_set = []

    # Add the sequence to all of these entries!
    for ec_num in enzymes_data:
        for org_num in enzymes_data[ec_num]:
            ec_org_code = f"{ec_num}_{org_num}"
            # ADD SEQUENCE ENRTY
            ref_ids = list(enzymes_data[ec_num][org_num]['ref_ids'])
            organism = enzymes_data[ec_num][org_num]['organism']


            # If we have no ref_ids, then we still need 1 None entry for this!
            if len(ref_ids) == 0:  
                ref_ids = [None]

            # Get the sequence entry
            # Check if the subs/prods are equiv not refs; if yes,
            # fuse the refs and comments.
            # ADD REACTION SMILES
            rxn_smiles = []
            # Add compounds to all these
            for rxn_header in REACTION_HEADERS:
                for rxn in enzymes_data[ec_num][org_num].get(rxn_header, []):
                    substrate_smiles = extract_smiles(rxn["DESC"]["SUBSTRATES"])
                    product_smiles = extract_smiles(rxn["DESC"]["PRODUCTS"])
                    refs = rxn["REFS"]
                    comments = rxn["COMMENTS"]
                    uniq_smiles = set(substrate_smiles).union(
                        set(product_smiles))

                    # New entry: 
                    entry = {
                        "SUBSTRATES": substrate_smiles,
                        "PRODUCTS": product_smiles,
                        "REFS": list(refs),
                        "COMMENTS": comments, 
                        "RXN_TEXT": " = ".join(
                            [" + ".join(rxn["DESC"]["SUBSTRATES"]), 
                             " + ".join(rxn["DESC"]["PRODUCTS"])]
                        )
                    }

                    # Make sure there's not only 1 smiles (unk)
                    # Add entry to rxn smiles IFF it's unique in subs and prods 
                    # for this org num and ec class
                    if len(uniq_smiles) > 1:
                        update_refs(rxn_smiles, entry, is_equal_rxn)

            # Export these to a dataset
            # These should be given to each ref id; if no ref id, give it to
            # None
            for ref_id in ref_ids: 
                for rxn_smile in rxn_smiles:
                    obj = {
                        "EC_NUM": ec_num,
                        "ORG": organism, 
                        "ORG_NUM": org_num,
                        "ORG_EC_KEY": ec_org_code,
                        "SEQ_ID": ref_id,
                        "REFS": " ".join(rxn_smile["REFS"]),
                        "COMMENTS": concat_comments(rxn_smile["COMMENTS"]),
                        "PRODUCTS": ".".join(rxn_smile["PRODUCTS"]),
                        "SUBSTRATES": ".".join(rxn_smile["SUBSTRATES"]), 
                        "RXN_TEXT" : rxn_smile["RXN_TEXT"]
                    }
                    rxn_set.append(obj)


            # Now let's handle the compounds
            enzymes_data[ec_num][org_num]['RXN_SMILES'] = rxn_smiles

            # Starting compound
            for cmp_header in COMPOUND_HEADERS:
                new_header = f"{cmp_header}_SMILES"
                compound_smiles = []
                for compound in enzymes_data[ec_num][org_num].get(
                        cmp_header, []):
                    smiles = mapped_compounds.get(compound["DESC"], None)
                    refs = compound["REFS"]
                    comments = compound["COMMENTS"]
                    entry = {
                        "SMILES": smiles,
                        "REFS": list(refs),
                        "COMMENTS": comments, 
                        "TEXT": compound["DESC"]
                    }

                    if smiles:
                        update_refs(compound_smiles, entry, is_equal_comp)

                enzymes_data[ec_num][org_num][new_header] = compound_smiles

                # Export these to a list of dicts that will be turned into a
                # csv
                for seq  in ref_ids: 
                    for compound in compound_smiles:
                        obj = {
                            "EC_NUM": ec_num,
                            "ORG_NUM": org_num,
                            "ORG": organism, 
                            "SEQ_ID" : ref_id, 
                            "ORG_EC_KEY": ec_org_code,
                            "COMPOUND": compound["SMILES"],
                            "COMPOUND_TEXT": compound["TEXT"],
                            "REFS": " ".join(compound["REFS"]),
                            "COMMENTS": concat_comments(compound["COMMENTS"]),
                            "COMPOUND_TYPE": cmp_header
                        }
                        compound_set.append(obj)

    return enzymes_data, rxn_set, compound_set

def reverse_mapping(smiles_dict : dict, outfile : str = None) -> dict: 
    """reverse_mapping.

    Go from mol name to smiles to smiles to mol name list 
    Args:
        smiles_dict (dict): smiles_dict
        outfile (str): Name of outfile

    Returns:
        dict:
    """
    ret_dict = defaultdict(lambda : [])
    for comp_name, smiles in smiles_dict.items(): 
        ret_dict[smiles].append(comp_name)

    utils.dump_json(ret_dict, outfile)
    return ret_dict

##### New functions #####
        
def get_args():
    """Get arguments"""
    options = argparse.ArgumentParser()

    options.add_argument('--brenda-flat-file',
                         action="store",
                         help="""BRENDA flat file as downloaded from their
                         website""",
                         required=True)
    options.add_argument('--brenda-ligands',
                         action="store",
                         help="""BRENDA flat file as downloaded from their
                         website""",
                         default= None)
    options.add_argument('--uniprot-download-folder',
                         action="store",
                         type=str,
                         help="Location for storing saved uniprot queries",
                         default="data/raw/Uniprot")
    options.add_argument('--out-prefix',
                         action="store",
                         help="""BRENDA parsed output directory prefix; will
                         store pasred file, gene list, and compound list
                         here""",
                         required=True)
    options.add_argument('--debug',
                         action="store_true",
                         default=False,
                         help="If true, early stop and debug")
    options.add_argument('--load-prev',
                         action="store_true",
                         default=False,
                         help="If true, try to load previous runs of program")
    options.add_argument('--use-cirpy',
                         action="store_true",
                         default=False,
                         help="If true, use cirpy to resolve (long time)")
    options.add_argument('--multiprocess-num',
                         action="store",
                         type=int, 
                         default=1,
                         help="If greater than 1, multiprocess")
    options.add_argument("--standardizer-log",
                         action="store",
                         default=None,
                         help="""Running dictionary mapping query names to
                         returned smiles""")
    options.add_argument("--cirpy-log",
                         action="store",
                         default="cirpy_log.txt",
                         help="""Running dictionary mapping query names to
                         returned smiles""")
    options.add_argument("--no-standardize",
                         action="store_true",
                         default=False,
                         help="""If this flag is set, do not standardize the
                         smiles strings.""")
    options.add_argument("--opsin-loc",
                         action="store",
                         default=None,
                         help="""Loc for opsin.jar file; if this is None, then
                         opsin will not be used""")
    args = options.parse_args()
    return args

def setup_logs(args : argparse.Namespace, out_prefix : str): 
    """ Setup the parser and create the output file
    Args: 
        args (argparse.Namespace): Arguments parsed
        out_prefix (str): Name of output prefix

    """
    # Set up logger
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s', 
                        handlers=[logging.StreamHandler(sys.stdout), 
                                  logging.FileHandler(f'{args.out_prefix}_{int(time.time())}.log')]

                        )

    logging.info(f"Args: {str(args)}")

    # Make outfile if it doesn't exist
    out_folder = os.path.dirname(args.out_prefix)
    os.makedirs(out_folder, exist_ok=True)



def parse_ligands_file(brenda_ligands_out_file : str, 
                       brenda_ligands_in_file : str, load_prev : bool, 
                       debug : bool) -> dict: 
    """ parse_ligands_file. 
    
    Parse the Brenda Ligands file. If we're loading previous to save time, load
    it. 

    Args:
        brenda_ligands_out_file (str): Name of outfile
        brenda_ligands_in_file (str): Name of ligands in file
        load_prev (bool) : If true, try to load from a prevoius parse
        deubg (bool): If true debug

    Return:
        Brenda ligands dict mapping common name to chebi and inchi 
    """

    # Parse brenda ligands
    if load_prev and os.path.exists(brenda_ligands_out_file):
        brenda_ligands = utils.load_json(brenda_ligands_out_file)
    else:
        brenda_ligands = parse_brenda_flatfile.parse_brenda_ligand(brenda_ligands_in_file)
        utils.dump_json(brenda_ligands,
                        brenda_ligands_out_file,
                        pretty_print=False)
    if debug: 
        import random
        keys_to_use = random.sample(brenda_ligands.keys(), k = int(0.01 * len(brenda_ligands)))
        new_dict = {i : brenda_ligands[i]  
                    for i in keys_to_use}
        
    return brenda_ligands

def parse_flat_file(ec_stats_out_file: str, enzymes_data_out_file : str, 
                    brenda_flat_file: str, load_prev : bool, 
                    out_prefix : str, debug : bool = False) -> Tuple[dict, dict]: 
    """ parse_flat_file. 
    
    Parse the Brenda Ligands file. If we're loading previous to save time, load
    it. 

    Args:
        ec_stats_out_file (str): Name of outfile
        enzymes_data_out_file (str): Name of outfile
        brenda_flat_file (str): Name of brenda flat file
        load_prev (bool) : If true, try to load from a prevoius parse
        out_prefix (str) : Out prefix used for logging
        debug (bool): If true, debug

    Return:
        Tuple[dict,dict] : ec_stats, enzymes_data
    """

    if load_prev and os.path.exists(ec_stats_out_file) and os.path.exists(
            enzymes_data_out_file):
        ec_stats = utils.load_json(ec_stats_out_file)
        enzymes_data = utils.load_json(enzymes_data_out_file)
    else:
        ec_stats, enzymes_data = parse_brenda_flatfile.read_brenda_file(
            brenda_flat_file, out_prefix, debug)
        utils.dump_json(ec_stats, ec_stats_out_file, pretty_print=False)
        utils.dump_json(enzymes_data,
                        enzymes_data_out_file,
                        pretty_print=False)
    return ec_stats, enzymes_data

def get_compound_list(enzymes_data : dict, compound_list_out: str, 
                      load_prev : bool) -> list: 
    """ get_compound_list. 
    
    Try to extract the compound list from the enzymes data. 
    If already exists, load it.

    Args:
        enzymes_data (dict): Parsed enzyme data
        compound_list_out (str): Name of outfile
        load_prev (bool) : If true, try to load from a prevoius parse

    Return:
        set: containing all compound strings for reactants, products,
        inhibitors, and activators
    """

    # Extract all compounds to resolve from brenda enzyme parse
    if load_prev and os.path.exists(compound_list_out):
        compound_list = utils.load_json(compound_list_out)
    else:
        # Store all compounds
        compound_list = set()

        for ec_class in enzymes_data:
            enzyme_data = enzymes_data[ec_class]
            # Extract reactions, inhibitors, and ID's
            for k, v in enzyme_data.items():

                # Iterate over all reactions
                for rxn_header in REACTION_HEADERS:
                    for rxn in v.get(rxn_header, []):
                        compound_list.update(
                            [compound for compound in rxn["DESC"]["PRODUCTS"]])
                        compound_list.update(
                            [compound for compound in rxn["DESC"]["SUBSTRATES"]])
                # Iterate over all containing compounds
                for compound_header in COMPOUND_HEADERS:
                    compound_list.update([
                        comp["DESC"]
                        for comp in v.get(compound_header, [])
                        if comp["DESC"].strip()
                    ])
                # Iterate over all kinetic items (e.g. Km)
                for kinetic_header in KINETIC_HEADERS: 
                    for kinetic_entry in v.get(kinetic_header, []): 
                        compound_list.update(
                            [comp for comp in kinetic_entry.get("SUB", []) if comp.strip()])

        utils.dump_json(list(compound_list),
                             compound_list_out,
                             pretty_print=False)
    return compound_list


def extract_unique_inchi_chebi(chebi_inchi_set_file : str, brenda_ligands: dict,
                               load_prev : bool,) -> dict:
    """extract_unique_inchi_chebi.

    Args:
        chebi_inchi_set_file (str): Name of file
        brenda_ligands (dict): brenda_ligands
        load_prev (bool): If true, try to load this from previous run 


    Returns:
        dict: {"inchi": inchi_set, "chebi" : chebi_set}
    """

    # Make set of inchis and set of CheBI's from brenda ligand
    # Use these for external query for identifier dict
    if load_prev and os.path.exists(chebi_inchi_set_file):
        chebi_inchi_set = utils.load_json(chebi_inchi_set_file)
    else:
        inchi_set = set()
        chebi_set = set()
        for k in brenda_ligands:
            inchi = brenda_ligands[k]["inchi"]
            chebi = brenda_ligands[k]["chebi"]
            if inchi:
                inchi_set.add(inchi)
            if chebi:
                chebi_set.add(chebi)
        chebi_inchi_set = {"inchi_set": list(inchi_set), "chebi_set": list(chebi_set)}
        utils.dump_json(chebi_inchi_set,
                             chebi_inchi_set_file,
                             pretty_print=False)
    return chebi_inchi_set



def map_chebi_to_smiles(chebi_inchi_set : dict,
                        mapped_chebi_file :str,
                        unmapped_chebi_file : str, 
                        load_prev : bool,
                        out_prefix: str, 
                        use_cirpy : bool, 
                        cirpy_log : str) -> Tuple[dict, list]:
    """map_chebi_to_smiles.

    Args:
        chebi_inchi_set (dict): chebi_inchi_set
        mapped_chebi_file (str): mapped_chebi_file
        unmapped_chebi_file (str): unmapped_chebi_file
        load_prev (bool): load_prev
        out_prefix (str): out_prefix
        use_cirpy (bool): use_cirpy
        cirpy_log (str): cirpy_log

    Returns:
        dict, list: Mapped chebi to smiles and unmapped chebi to smiles
    """
    if (load_prev and os.path.exists(mapped_chebi_file) and
            os.path.exists(unmapped_chebi_file)):
        mapped_chebi = utils.load_json(mapped_chebi_file)
        unmapped_chebi = utils.load_json(unmapped_chebi_file)
    else:

        mapped_chebi, unmapped_chebi = parse_brenda_compounds.chebi_to_smiles(
            list(chebi_inchi_set["chebi_set"]), out_prefix, use_cirpy,
            cirpy_log)
        utils.dump_json({i: list(j) for i, j in mapped_chebi.items()},
                             mapped_chebi_file,
                             pretty_print=False)
        utils.dump_json(list(unmapped_chebi),
                             unmapped_chebi_file,
                             pretty_print=False)
    return mapped_chebi, unmapped_chebi

def map_inchi_to_smiles(chebi_inchi_set : dict, mapped_inchi_file :str,
                        unmapped_inchi_file : str, load_prev : bool,
                        out_prefix: str, use_cirpy : bool, 
                        cirpy_log : str) -> Tuple[dict, list]:
    """map_chebi_to_smiles.

    Args:
        chebi_inchi_set (dict): chebi_inchi_set
        mapped_inchi_file (str): mapped_chebi_file
        unmapped_inchi_file (str): unmapped_chebi_file
        load_prev (bool): load_prev
        out_prefix (str): out_prefix
        use_cirpy (bool): use_cirpy
        cirpy_log (str): cirpy_log

    Returns:
        dict, list: Mapped chebi to smiles and unmapped chebi to smiles
    """

    if (load_prev and os.path.exists(mapped_inchi_file) and
            os.path.exists(unmapped_inchi_file)):
        mapped_inchi = utils.load_json(mapped_inchi_file)
        unmapped_inchi = utils.load_json(unmapped_inchi_file)
    else:

        mapped_inchi, unmapped_inchi = parse_brenda_compounds.inchi_to_smiles(
            list(chebi_inchi_set["inchi_set"]), out_prefix, use_cirpy,
            cirpy_log)
        utils.dump_json({i: list(j) for i, j in mapped_inchi.items()},
                             mapped_inchi_file,
                             pretty_print=False)
        utils.dump_json(list(unmapped_inchi),
                             unmapped_inchi_file,
                             pretty_print=False)
    return mapped_inchi, unmapped_inchi


def map_compounds_to_smiles(compound_list : dict, brenda_ligands : list,
                            mapped_inchi : dict, mapped_chebi : dict,
                            mapped_comps_file : str, unmapped_comps_file: str,
                            load_prev : bool, out_prefix: str, 
                            use_cirpy : bool, cirpy_log : str, 
                            opsin_loc : str) -> Tuple[dict, list]:
    """map_compounds_to_smiles.

    Args:
        compound_list (dict): List of compounds to resolve
        brenda_ligands (dict): Parsed brenda ligand file
        mapped_inchi (dict): Dict of mapped inchi values
        mapped_chebi (dict): Dict of mapped chebi values
        chebi_inchi_set (dict): chebi_inchi_set
        mapped_comps_file (str): Out file
        unmapped_comps_file (str): Out file
        load_prev (bool): load_prev
        out_prefix (str): out_prefix
        use_cirpy (bool): use_cirpy
        cirpy_log (str): cirpy_log
        opsin_loc (str): Location of opsin 

    Returns:
        dict, list: Mapped compounds to smiles and unmapped compounds to smiles
    """

    if (load_prev and os.path.exists(mapped_comps_file) and
            os.path.exists(unmapped_comps_file)):
        mapped_compounds = utils.load_json(mapped_comps_file)
        unmapped_compounds = utils.load_json(unmapped_comps_file)
    else:
        mapped_compounds, unmapped_compounds = parse_brenda_compounds.resolve_compound_list(
            compound_list, brenda_ligands, mapped_inchi, mapped_chebi,
            out_prefix, use_cirpy, cirpy_log, opsin_loc)
        utils.dump_json(mapped_compounds, mapped_comps_file)
        utils.dump_json(unmapped_compounds, unmapped_comps_file)
    return mapped_compounds, unmapped_compounds


def standardize_smiles(mapped_standardized_file : str, mapped_compounds : dict,
                       standardizer_log : str, load_prev : bool, 
                       multiprocess_num : int) -> dict: 
    """standardize_smiles.

    Args:
        mapped_standardized_file (str): mapped_standardized_file
        mapped_compounds (dict): mapped_compounds
        standardizer_log (str): standardizer_log
        load_prev (bool): load_prev
        multiprocess_num (int): multiprocess_num

    Returns:
        dict: Standardized smiels
    """
    if load_prev and os.path.exists(mapped_standardized_file): 
        mapped_compounds = utils.load_json(mapped_standardized_file)
    else: 
        mapped_compounds = parse_brenda_compounds.standardize_mols(mapped_compounds, 
                                                                   standardizer_log=standardizer_log, 
                                                                   multiprocess_count = multiprocess_num)
        # Dump outputs
        utils.dump_json(mapped_compounds, mapped_standardized_file)
    return mapped_compounds

def main():
    """ Main method to run this parse """
    args = get_args()
    setup_logs(args, args.out_prefix)


    # Parse Brenda Ligands file
    logging.info("Starting to parse ligands")
    brenda_ligands_file = f"{args.out_prefix}_brenda_ligands.json"
    brenda_ligands = parse_ligands_file(brenda_ligands_file,
                                        args.brenda_ligands, args.load_prev,
                                        args.debug)
    logging.info("Done parsing ligands")

    # Parse brenda ec classes and enzymes
    ec_stats_file = f"{args.out_prefix}_brenda_ec_stats.json"
    enzymes_data_file = f"{args.out_prefix}_brenda_enzymes_data.json"

    # Parse brenda flat file
    # Ec_stats contains enzyme class wide parameters
    # enzymes_data contains actual enzymes
    logging.info("Starting to parse flat file")
    ec_stats, enzymes_data = parse_flat_file(ec_stats_file, 
                                             enzymes_data_file, 
                                             args.brenda_flat_file,
                                             args.load_prev,
                                             args.out_prefix,
                                             args.debug)
    logging.info("Done parsing flat file")

    # Get the list of compounds from enzymes data 
    logging.info("Starting to extract compound list")
    compounds_list_file = f"{args.out_prefix}_compound_list.json"
    compound_list = get_compound_list(enzymes_data, compounds_list_file, 
                                      args.load_prev)
    logging.info("Done extracting compound list")

    logging.info("Starting to extract inchi/chebi")
    chebi_inchi_set_file = f"{args.out_prefix}_chebi_inchi_set.json"
    chebi_inchi_set = extract_unique_inchi_chebi(chebi_inchi_set_file, 
                                                 brenda_ligands,
                                                 args.load_prev)
    logging.info("Done extracting inchi/chebi")

    logging.info("Starting to map chebi to smiles")
    # Map brenda ligand chebi to smiles
    mapped_chebi_file = f"{args.out_prefix}_chebi_to_smiles.json"
    unmapped_chebi_file = f"{args.out_prefix}_chebi_unmapped.json"
    mapped_chebi, unmapped_chebi = map_chebi_to_smiles(chebi_inchi_set, 
                                                       mapped_chebi_file,
                                                       unmapped_chebi_file, 
                                                       args.load_prev,
                                                       args.out_prefix,
                                                       args.use_cirpy,
                                                       args.cirpy_log)
    logging.info("Done mapping chebi to smiles")

    logging.info("Starting to map inchi to smiles")
    # Map brenda ligand inchi to smiles
    mapped_inchi_file = f"{args.out_prefix}_inchi_to_smiles.json"
    unmapped_inchi_file = f"{args.out_prefix}_inchi_unmapped.json"
    mapped_inchi, unmapped_inchi = map_inchi_to_smiles(chebi_inchi_set,
                                                       mapped_inchi_file,
                                                       unmapped_inchi_file,
                                                       args.load_prev,
                                                       args.out_prefix,
                                                       args.use_cirpy,
                                                       args.cirpy_log)
    logging.info("Done mapping inchi to smiles")

    logging.info("Starting to resolve all compounds to smiles")
    # Now resolve all compounds
    mapped_comps_file = f"{args.out_prefix}_compounds_to_smiles.json"
    unmapped_comps_file = f"{args.out_prefix}_compounds_unmapped.json"

    # TODO: Fix this so that it doesn't return strings and lists, but only one
    # of the two types 
    mapped_compounds, unmapped_compounds = map_compounds_to_smiles(compound_list, brenda_ligands, 
                                                                   mapped_inchi, mapped_chebi,
                                                                   mapped_comps_file, unmapped_comps_file,
                                                                   args.load_prev, args.out_prefix,
                                                                   args.use_cirpy, args.cirpy_log, 
                                                                   args.opsin_loc) 
    logging.info("Done resolving all compounds to smiles")


    # Standardize all smiles mappings!
    # Load from file if it exists
    if not args.no_standardize:
        logging.info("Starting standardizer")
        mapped_standardized_file = f"{args.out_prefix}_compounds_to_standardized_smiles.json"
        mapped_compounds = standardize_smiles(mapped_standardized_file, mapped_compounds,
                                              args.standardizer_log, args.load_prev, 
                                              args.multiprocess_num)
        logging.info("Done with standardizer")


    logging.info("Beginning to export all files")
    # Reverse the mapping for reference later 
    smiles_to_names_file = f"{args.out_prefix}_smiles_to_comp_names.json"
    smiles_to_names = reverse_mapping(mapped_compounds, 
                                      outfile = smiles_to_names_file)

    # Add gene sequences and compound smiles back to the dataframe.
    enzymes_data, rxn_set, compound_set = add_compounds(
        enzymes_data, mapped_compounds)
    rxn_set, compound_set = pd.DataFrame(rxn_set), pd.DataFrame(compound_set)

    # Ouput finished mapping to file
    enzymes_mapped_data_file = f"{args.out_prefix}_enzymes_data_complete_mapped.json"
    rxn_final_tsv = f"{args.out_prefix}_rxn_final.tsv"
    compounds_final_tsv = f"{args.out_prefix}_rxn_compounds.tsv"
    utils.dump_json(enzymes_data,
                    enzymes_mapped_data_file,
                    pretty_print=False)

    compound_set.to_csv(compounds_final_tsv, sep="\t")
    rxn_set.to_csv(rxn_final_tsv, sep="\t")

    # Output statistics about all the data collected
    summary_file = f"{args.out_prefix}_stats.json"
    stats_summary = {
        "mapped_compound_smiles": len(mapped_compounds),
        "unmapped_compound_smiles": len(unmapped_compounds),
        "mapped_inchis": len(mapped_inchi),
        "unmapped_inchis": len(unmapped_inchi),
        "mapped_chebi": len(mapped_chebi),
        "unmapped_chebi": len(unmapped_chebi),
    }

    stats_summary["Num ec classes"] = len(ec_stats)

    # Update with other statistics and print this to a file
    stats_summary.update(
        parse_brenda_stats.get_rxn_compound_stats(rxn_set, compound_set)
    )
    utils.dump_json({k: str(v) for k, v in stats_summary.items()},
                         summary_file)

