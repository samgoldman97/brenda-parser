""" parse_brenda_flatfile.py

This script stores functions necessary to parse explicitly the brenda flatfile
into a complete dictionary. This is called by a master parse script, which
combines information from genes and compound attribution.

read_brenda_file is the main function meant to be exposed.
"""

import re
import os
import typing
from typing import Tuple, Dict, Optional, Callable, List
from collections import defaultdict
import logging
import time
import numpy as np

from tqdm import tqdm 

# EC RE
EC_RE = r"\d+.\d+.\d+.\d+"
UNIPROT_RE = r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
GENBANK_RE = r"[A-Z]{1,6}[0-9]{5,8}"
PROTEIN_DB_RE = r"(UniProt|GenBank|SwissProt)"
HEADER_RE = r"^\w+$"
TAG_RE = r"(^(?:[A-Z]|[0-9])+)(?:\t|$|_)"
ORG_RE = r"^(#(?:[0-9]+[, ]*?)+#)"  # some are misannotated w/ spaces
LIST_SPLIT_RE = r"[, ]"  # Split lists at commas or spaces
REF_ID_RE = r"^(<.*?>)"  # For reference id matches
REF_RE = r"^.*(<.*?>)[^<>]*$"  # Get LAST reference in the string
COMMENT_RE = r"(\({}.*\))".format(ORG_RE.replace(
    "^", ""))  # Enforce comment RE must have a ref at start
PRODUCT_COMMENT_RE = r"(\|{}.*\|)".format(ORG_RE.replace(
    "^", ""))  # Enforce comment RE must have a ref at start
COMMENT_SHELL_RE = r"^[\|\(](.+)[\|\)]$"  # Extract comment boundaries
COMMENT_SEP = r";"


SUBS_RE = r"([AC-IK-NP-TVWY])([0-9]+)([AC-HK-NP-TVWY]|I(?!ns))" # E.g. A212V/C220M substitutions; avoid C220InsA
INS_RE = "([AC-IK-NP-TVWY])([0-9]+)Ins([AC-IK-NP-TVWY])" # Capture insertions
INS_RE_2 = "ins([0-9]+)([AC-IK-NP-TVWY])" 
INS_RE_3 = "([AC-IK-NP-TVWY])([0-9]+) insertion"
DEL_RE= r"(?<!\w)del\s{0,1}([AC-IK-NP-TVWY])([0-9]+)"
NO_ACTIVITY_RE = r"no activity in"

DEFAULT_PROT_DB = r"UniProt"  # Have a default in case the id is present but not the db
REACTION_HEADERS = [
    "REACTION", "SUBSTRATE_PRODUCT", "NATURAL_SUBSRTATE_PRODUCT"
]
COMPOUND_HEADERS = ["INHIBITORS", "ACTIVATING_COMPOUND", "COFACTOR"]
KINETIC_HEADERS = ["TURNOVER_NUMBER", "KM_VALUE"]

unique_headers = dict()
non_specific_headers = set()


def extract_orgs_desc(line: str) -> Tuple[list, str, dict, list]:
    """extract_orgs_desc.

    Extract organisms involved, description, comments, and references
    
    Example of a line being handled: 
        #3,7# 2-oxoglutarate + CoA + 2 oxidized ferredoxin = succinyl-CoA + CO2 + 2
        reduced ferredoxin + 2 H+ (#7# specific for 2-oxoglutarate <5>; #3# pure
        enzyme does not utilize other 2-keto acids, such as pyruvate,
        phenylglyoxylate, indolepyruvate, 2-oxobutanoate or 2-oxoisovalerate, as
        substrates <3>) <3,5>

        We want to pull out the organisms participating (#3,7#) and reduce the
        second part of the string to get rid of parantehtical and items

    Args:
        line (str): line

    Returns:
        Tuple[list, str, dict, list]: (orgs, desc, comments, refs)
    """

    # Format of line:
    # SY    #190# Pcal_1699 (#190# gene name <133>) <133>
    # First split to get number and parse that
    # Extract orgs
    org_match = re.match(ORG_RE, line)
    if org_match:
        orgs_str = org_match.group()
        orgs = [
            i.strip() for i in re.split(LIST_SPLIT_RE,
                                        org_match.group()[1:-1])
        ]
        # Replace rest of line
        line = re.sub(orgs_str, "", line, count=1)
    else:
        orgs = []

    # Extract references
    refs_match = re.search(REF_RE, line)
    if refs_match:
        refs_str = refs_match.groups()[0].strip()

        refs = [i.strip() for i in re.split(LIST_SPLIT_RE, refs_str[1:-1])]

        # Replace last occurence
        line = "".join(line.rsplit(refs_str, 1))
    else:
        refs = []

    # comments should be a dict with org num :
    # [("COMMENT" : desc, "REFS" : # [refs]), ("COMMENT" : desc, "REFS" : [refs])]
    com_dict = defaultdict(lambda: [])

    # Extract products comments
    prod_comments_match = re.search(PRODUCT_COMMENT_RE, line)
    if prod_comments_match:
        prod_comments = prod_comments_match.groups()[0]
        line = line.replace(prod_comments, "", 1)
        prod_comments = re.sub(COMMENT_SHELL_RE, r"\1", prod_comments)
        prod_comments = prod_comments.split(COMMENT_SEP)
        # Dangerous recursion
        prod_com_list = [extract_orgs_desc(j) for j in prod_comments]
        for com_orgs, com_desc, com_com, com_ref in prod_com_list:
            for com_org in com_orgs:
                com_dict[com_org].append({"COMMENT": com_desc, "REFS": com_ref})

    # else:
    #     prod_comments = ""

    # Now extract comments
    comments_match = re.search(COMMENT_RE, line)
    if comments_match:
        comments = comments_match.groups()[0]
        line = line.replace(comments, "", 1)
        comments = re.sub(COMMENT_SHELL_RE, r"\1", comments)
        comments = comments.split(COMMENT_SEP)
        # Dangerous recursion
        com_list = [extract_orgs_desc(j) for j in comments]
        for com_orgs, com_desc, com_com, com_ref in com_list:
            for com_org in com_orgs:
                com_dict[com_org].append({"COMMENT": com_desc, "REFS": com_ref})

    # else:
    #     comments = ""

    # Join comments and prod comments
    # comments = "".join([comments, prod_comments])
    desc = line.strip().replace("\n", " ")

    return (orgs, desc, com_dict, refs)


def extract_reaction(rxn: str) -> dict:
    """extract_reaction.

    Helper string to convert a reaction into its component substraters,
    products, and reversibility info 

    Args:
        rxn (str): rxn

    Returns:
        dict: Contains substrates, products, and reversibility
    """
    orig_rxn = rxn

    COMPOUNDS_RE = r" \+ "
    SUB_PROD_RE = r" = "
    REVERSIBLE_RE = r"\{\w*\} *$"

    reversibility = re.search(REVERSIBLE_RE, rxn)

    if reversibility:
        rxn = rxn.replace(reversibility.group(), "").strip()
        reversibility = reversibility.group()

    reversibility = (reversibility.strip()[1:-1]
                     if reversibility and len(reversibility) > 2 else "?")

    split_prod = re.split(SUB_PROD_RE, rxn, 1)
    if len(split_prod) == 1:
        substrates = split_prod[0]
        products = ""
    elif len(split_prod) == 2:
        substrates, products = split_prod
    else:
        raise Exception(f"Unexpected number of reaction components in {rxn}")

    substrates = [
        i.strip() for i in re.split(COMPOUNDS_RE, substrates) if i.strip()
    ]
    products = [
        i.strip() for i in re.split(COMPOUNDS_RE, products) if i.strip()
    ]

    ret_dict = {
        "SUBSTRATES": substrates,
        "PRODUCTS": products,
        "REVERSIBLE": reversibility
    }

    return ret_dict


def entry_lines(body: str, tag: str):
    """entry_lines.

    Helper iterator

    Args:
        body (str): Body of text 
        tag (str): Tag separating the body

    Return: 
        Iterator containing lines to be parsed 
    """
    # Helper function to replace line with spaces, then split at token
    split_at_token = lambda body, token: body.replace("\n\t", " ").split(token)

    lines = split_at_token(body, tag + "\t")

    for line in lines:
        line = line.strip()
        line = line.replace("\n", " ")

        if not line:
            continue

        yield line


def get_parser(header):
    """ Return function that should parse this header"""

    parser_list = {"PROTEIN": parse_protein, 
                   "REFERENCE": parse_reference,
                   "KM_VALUE" : parse_kinetic, 
                   "TURNOVER_NUMBER": parse_kinetic}
    if header in parser_list:
        return parser_list[header]
    else:
        return None

def parse_kinetic(body: str, enzymes: dict, tag: str, ec_num: str,
                  header : str, **kwargs) -> None:
    """parse_kinetic.

    Args:
        body (str): body
        enzymes (dict): enzymes
        tag (str) : Tag associated with the entry header (without the tab)
        ec_num (str): ec number to use
        header (str): Header value

    Returns:
        None:
    """

    # Generic parse
    for line in entry_lines(body, tag):

        # Check that this is specific to an enzyme number
        org_nums, desc, comments, refs = extract_orgs_desc(line)
        KINETIC_SUB_RE = r"(\{.*\})"
        RANGE_RE = "\d(\s*-\s*)\d"

        substrate = re.search(KINETIC_SUB_RE, desc)
        sub_brackets = substrate.group() if substrate else None
        if sub_brackets: 
            desc = desc.replace(sub_brackets, "").strip()

        # Capture the non bracketed part of substrate
        sub = sub_brackets[1:-1]

        # Find intermediate number
        range_sep = re.search(RANGE_RE, desc)
        range_sep = range_sep.groups()[0] if range_sep else None
        if range_sep:
            range_groups = desc.split(range_sep)
            if len(range_groups) != 2: 
                raise ValueError("Found unexpected number of range values in rate constant")

            desc = [float(i) for i in range_groups]
            # Often range
            #if rate_groups[1] / rate_groups[0] > 10: 
        else: 
            # Convert rate to list
            desc = [desc]

        # If the header contains reaction data, use standard rxn
        entry = {"REFS": refs, "DESC": desc,"SUB": sub, "COMMENTS": comments}

        # If we have a new tag specific to this enzyme, add it to keep track!
        for org_num in org_nums:

            # Should already be added from protein entries
            if org_num not in enzymes:
                logging.warning((f"Org number {org_num} not "
                                 f"found in EC {ec_num} and "
                                 f"line {line} "))
                continue

            # Extract comments that belong only to this org!
            copy_entry = entry.copy()
            copy_entry["COMMENTS"] = copy_entry["COMMENTS"].get(
                org_num, [])

            enzymes[org_num][header].append(copy_entry)

def parse_reference(body: str, enzymes: dict, tag: str, ec_num: str,
                    general_stats: dict, **kwargs) -> None:
    """parse_reference.

    Args:
        body (str): body
        enzymes (dict): enzymes
        tag (str) : Tag associated with the entry header (without the tab)
        ec_num (str): ec number to use
        general_stats (dict): Dict containing stats for most items 

    Returns:
        None:
    """

    # Add to general_stats:
    if "REFERENCE" not in general_stats:
        general_stats["REFERENCE"] = defaultdict(lambda: "")

    for line in entry_lines(body, tag):

        # Extract references
        refs_match = re.search(REF_ID_RE, line)
        if refs_match:
            refs_str = refs_match.groups()[0].strip()
            ref_id = [i.strip() for i in refs_str[1:-1].split(",")][0]

            # Replace last occurence
            line = line.replace(refs_str, "", 1)
            line = "".join(line.rsplit(refs_str, 1))
            general_stats["REFERENCE"][ref_id] = line.strip()
        else:
            logging.warning(f"No ref found in entry {line} for ec {ec_num}")
            continue


def parse_protein(body: str, enzymes: dict, tag: str, ec_num: str,
                  **kwargs) -> None:
    """parse_protein.

    Args:
        body (str): body
        enzymes (dict): enzymes
        tag (str) : Tag associated with the entry header (without the tab)
        ec_num (str): ec number to use

    Returns:
        None:
    """
    # Split at thet tag
    for line in entry_lines(body, tag):
        org_num, pr_split, comments, refs = extract_orgs_desc(line)

        # Extract single org num
        if len(org_num) != 1:
            raise Exception(
                f"Found multiple organism numbers for protein {line}")
        else:
            org_num = org_num[0]

        # Search line for uniprot regex
        # Add support for multiple ref ids

        ref_ids = re.findall(UNIPROT_RE, pr_split)
        protein_db_default = "uniprot"

        # If empty set 
        if not ref_ids:
            ref_ids = re.findall(GENBANK_RE, pr_split)
            protein_db_default = "genbank"

        # extract which protein database
        protein_db = re.search(PROTEIN_DB_RE, pr_split, re.IGNORECASE)
        protein_db = protein_db.group().lower() if protein_db else protein_db

        # If protein db wasn't listed but we found an RE match for a ref id,
        # set it
        if not protein_db and ref_ids:   
            protein_db = protein_db_default

        # Handle no activity case
        no_activity = re.search(NO_ACTIVITY_RE, pr_split, re.IGNORECASE)
        no_activity = no_activity.group() if no_activity else no_activity
        is_negative = True if no_activity else False

        # Extract all extra categories
        if ref_ids:
            for ref_id in ref_ids: 
                pr_split = pr_split.replace(ref_id, "").strip()

            # Also replace the "and" token in case we have joiners
            pr_split = re.sub(" and ", "", pr_split, flags=re.IGNORECASE).strip()

        if protein_db:
            # Case insensitive!
            pr_split = re.sub(protein_db, "", pr_split, flags=re.IGNORECASE).strip()
        if no_activity:
            pr_split = pr_split.replace(no_activity, "").strip()

        organism = pr_split
        # Because we're defining organism numbers in this loop, comments must
        # be specific
        comments = comments.get(org_num, [])
        enzymes[org_num] = defaultdict(lambda: [])
        enzymes[org_num].update({
            "ec_num": ec_num,
            "organism": organism,
            "ref_ids": ref_ids,
            "protein_db": protein_db,
            "no_activity": is_negative,
            "refs": refs,
            "comments": comments,
        })

def process_entry(brenda_entry: str) -> Tuple[dict, Tuple[str, defaultdict]]:
    """process_entry.

    Args:
        brenda_entry (str): brenda_entry

    Returns:
        ec_num (str): EC number
        enzymes (dict): Mapping of enzymes to their parsed information
        general_stats (defaultdict): general stats for the ec class
        
    """
    ## Get ec number!
    ec_num = re.search(EC_RE, brenda_entry).group()

    # If we have any missing entries, it'll appear as \n\n\n, just mask these as \n\n to skip over
    # Sometimes a certain entry is missing and that throws off our parser.
    brenda_entry = re.sub(r"\n\n+", "\n\n", brenda_entry).strip()

    # Split into groups
    cats = brenda_entry.split("\n\n")

    # These will be shared across all dictionaries, so make one list up top
    # such that we don't need to replicate them
    enzymes = dict()

    # Dict to hold enzyme class general items
    general_stats = defaultdict(lambda: [])

    # Extra args:
    extra_args = {"ec_num": ec_num, "general_stats": general_stats}

    # Each brenda category should be a group in cats
    for j in cats:
        j = j.strip()
        header, body = j.split("\n", 1)
        header, body = header.strip(), body.strip()
        if not re.match(HEADER_RE, header):
            continue
        else:
            # Keep track of valid headers
            tag = re.search(TAG_RE, body)
            if not tag:
                raise Exception(
                    f"Unable to find tag for header {header} in {body}")
            elif (header in unique_headers and
                  unique_headers[header] != tag.groups(1)[0]):
                raise Exception(f"""Duplicate tag found for header {header}: 
                                {tag.groups(1)[0]}, {unique_headers[header]}""")
            else:
                unique_headers[header] = tag.groups(1)[0]

        # Parse header
        parse_fn = get_parser(header)
        tag = unique_headers[header]

        # If this tag should be parsed differently
        if parse_fn:
            parse_fn(body, enzymes, tag, header=header, **extra_args)
        else:

            # Generic parse
            for line in entry_lines(body, tag):

                # Check that this is specific to an enzyme number
                org_nums, desc, comments, refs = extract_orgs_desc(line)

                # If the header contains reaction data, use standard rxn
                if header in REACTION_HEADERS:
                    desc = extract_reaction(desc)

                entry = {"REFS": refs, "DESC": desc, "COMMENTS": comments}

                # If we have a new tag specific to this enzyme, add it to keep track!
                for org_num in org_nums:

                    # Should already be added from protein entries
                    if org_num not in enzymes:
                        logging.warning((f"Org number {org_num} not "
                                         f"found in EC {ec_num} and "
                                         f"line {line} "))
                        continue

                    # Extract comments that belong only to this org!
                    copy_entry = entry.copy()
                    copy_entry["COMMENTS"] = copy_entry["COMMENTS"].get(
                        org_num, [])

                    enzymes[org_num][header].append(copy_entry)

                # If line doesn't have an org number and is GENERAl to EC class
                if not org_nums:
                    general_stats[header].append(entry)

    return ec_num, enzymes, general_stats


def distribute_ref_ids(enzymes_data : dict) -> int: 
    """distribute_ref_ids.

    Assign enzyme ids to unlabeled entries of the same organism

    Args:
        enzymes_data (dict): enzymes_data

    Returns:
        int: Number of reassignments made
    """

    num_reassigned = 0
    for ec_num in enzymes_data: 
        # Map {organism : set([(ref_id, protein_db), (ref_id, protein_db), ...] )}
        org_to_ref_id = defaultdict(lambda : set())
        ref_ids_to_refs = defaultdict(lambda : set())
        for org_num in enzymes_data[ec_num]: 
            organism = enzymes_data[ec_num][org_num].get("organism", None)
            ref_ids = enzymes_data[ec_num][org_num].get("ref_ids", [])
            protein_db = enzymes_data[ec_num][org_num].get("protein_db", None)
            refs = enzymes_data[ec_num][org_num].get("refs", None)

            organism = organism.lower() if organism is not None else organism 

            # Continue if no organism
            if organism is None or not ref_ids or protein_db is None: 
                continue
            else: 
                for ref_id in ref_ids: 
                    org_to_ref_id[organism].add((ref_id, protein_db))
                    # Add all references into this!
                    ref_ids_to_refs[ref_id].update(refs)

        # Now reattribute by looping again!
        for org_num in enzymes_data[ec_num]: 
            organism = enzymes_data[ec_num][org_num].get("organism", None)
            ref_ids = enzymes_data[ec_num][org_num].get("ref_ids", [])
            organism = organism.lower() if organism is not None else organism 

            # Continue if already has an annotation
            if ref_ids: 
                continue
            # If this has no ref ID _AND_ this organism was seen elsewhere
            elif organism in org_to_ref_id: 
                ref_ids = org_to_ref_id[organism]
                if len(ref_ids) != 1: 
                    logging.warning(f"Found {len(ref_ids)} protein ref ids for organism {organism} in ec class {ec_num}")

                # Normally lowest to highest, we want to go highest refs to lowest!
                sorted_ref_ids = sorted(ref_ids, key = lambda x: len(ref_ids_to_refs[x[0]]))[::-1]

                #print(f"Top number of refs: {len(ref_ids_to_refs[sorted_ref_ids[0][0]])}")
                #print(f"Bottom number of refs: {len(ref_ids_to_refs[sorted_ref_ids[-1][0]])}")

                new_ref, new_db = sorted_ref_ids[0]

                # Give first id!
                enzymes_data[ec_num][org_num]["ref_ids"] = [new_ref]
                # If there was a reassignment, we should acknowledge this isn't

                # Show that the original annotation was an empty list (or if it
                # was set already, kep that) 
                enzymes_data[ec_num][org_num]["seq_inferred"] = True

                num_reassigned += 1
            else: 
                # If this entry has no uniprot accession but also no others listed
                continue

    return num_reassigned

def read_brenda_file(in_file: str,
                     out_prefix: str = "temp",
                     debug: bool = False) -> Tuple[dict, dict]:
    """read_brenda_file.

    Args:
        in_file (str): Name of flat BRENDA file
        out_prefix (str): out_prefix
        debug (bool): If true, stop early

    Returns:
        Tuple[dict,dict]:ec_stats, enzymes_data
    """

    enzymes_data = dict()
    ec_stats = dict()

    buff = ""

    num_lines, line_count = 0, 0
    with open(in_file, "r") as fp: 
        for line in fp: num_lines += 1

    pbar = tqdm(total=num_lines)
    with open(in_file, "r") as fp:
        new_line = fp.readline()
        while (new_line):
            # Get the next line if we have a blank
            if buff == "":
                buff += new_line
            else:
                # If we reach a new BRENDA header
                if r"///" in new_line:

                    # Process the Brenda entry
                    ec_num, enzyme_data, general_stats = process_entry(buff)

                    # Add to data log
                    ec_stats[ec_num] = general_stats
                    enzymes_data[ec_num] = enzyme_data

                    buff = new_line
                elif len(buff) > 0:
                    buff += new_line
                else:
                    raise RuntimeError("Unexpected parse logic")

            new_line = fp.readline()
            pbar.update()

            # For debugging
            if debug and len(ec_stats) > 2000:
                break
    pbar.close()

    return ec_stats, enzymes_data


def remove_umatched_uniprot_ids(enzymes_data : dict, 
                                found_ids : set) -> None: 
    """remove_umatched_uniprot_ids.

    Args:
        enzymes_data (dict): enzymes_data
        found_ids (set): found_ids

    Returns:
        None:
    """
    for ec_num in enzymes_data: 
        for org_num in enzymes_data[ec_num]: 
            ref_ids = enzymes_data[ec_num][org_num].get("ref_ids", [])

            # If we didn't find a sequence for this
            for ref_id in ref_ids and ref_id not in found_ids: 
                    #  remove ref id from the list of reference ids!
                    enzymes_data[ec_num][org_num]["ref_ids"].remove(ref_id)
##### Parsing ligand file

def parse_brenda_ligand(brenda_ligand_file: str) -> dict:
    """parse_brenda_ligand.

    Args:
        brenda_ligand_file (str): brenda_ligand_file

    Returns:
        dict: {name : {chebi: chebi,inchi : inchi}}
    """

    ret = {}
    if brenda_ligand_file: 
        with open(brenda_ligand_file, "rb") as fh:
            for line in tqdm(fh):
                try:
                    name, _, _, _, inchi, chebi = line.strip().decode(
                        'utf-8').split('\t')

                    chebi = None if len(chebi.strip()) == 1 else chebi
                    inchi = None if len(inchi.strip()) == 1 else inchi

                    # only add if chebi or inchi exists
                    if inchi or chebi:
                        ret[name.lower()] = {"chebi": chebi, "inchi": inchi}
                except UnicodeDecodeError:
                    logging.warning(f"Couldn't decode ligand {line}")
    else: 
        logging.warning(f"No file {brenda_ligand_file} found")

    return ret

