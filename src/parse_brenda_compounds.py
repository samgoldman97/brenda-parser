"""parse_brenda_compounds.py

This script contains functions intended to deidentify molecules for brenda

"""

import os
import subprocess
import multiprocessing
import typing
from typing import Tuple, Callable, List, Optional
import re
import cirpy
import numpy as np
import logging
from rdkit import Chem
import molvs
from collections import defaultdict

from src import utils

STOICH_RE = '^[0-9]+ +'
OPSIN_URL = "https://bitbucket.org/dan2097/opsin/downloads/opsin-2.4.0-jar-with-dependencies.jar"
TEMP_OPSIN_INPUT = ".opsin_temp_input.txt"
TEMP_OPSIN_OUTPUT = ".opsin_temp_output.txt"

def strip_stoich(mol: str) -> Tuple[int, str]:
    """ Remove stoichiometry prefix and return int, str"""
    # First, replace the number in the beginnin
    # check for multiple substrates or products, e.g. 2 NAD+
    stoich_coeff = re.search(STOICH_RE, mol)
    if stoich_coeff:
        stoich_coeff = int(stoich_coeff.group()[0].strip())
        mol = re.sub(STOICH_RE, '', mol)
    else:
        stoich_coeff = 1
    return stoich_coeff, mol

def get_substitution_rules() -> List[Callable[[str], str]]:
    """ Return different functions that can be applied, 

    Each function need not worry about returning none if the rule doesn't
    apply; this should be handleed in an outside loop

    Return: 
        List[Callable[[str], str]]: List of string substitution functions
    """

    def identity(x):
        """ Identity fn """
        return x

    def strip_stoich_wrapper(x):
        """ Remove stoich num from beginining """
        _, x = strip_stoich(x)
        return x

    def comma_for_space(x):
        """For when spaces in inhibitors got added bc of new lines"""
        x = strip_stoich_wrapper(x)
        x = x.replace(" ", ",")
        return x

    def remove_plus_minus(x):
        """ molecules with L (-) or R(+) information, remove (-) or (+) """
        x = strip_stoich_wrapper(x)
        REPLACE_GROUP = r'([LSRD])[ -](\([+-]\))'
        # Adding in the r'\1' ensures that we keep the L component, sub the rest
        new_x = re.sub(REPLACE_GROUP, r'\1', x)
        return new_x

    def swap_LS_DR(x):
        """ try swapping L<->S and D<->R for chirality """
        SWAP_MAP = {"L": "S", "S": "L", "D": "R", "R": "D"}

        # First capture group is optional left paren
        # Second capture group is LRSD
        # Third capture group is optional right paren
        REPLACE_GROUP = r'([\(]{0,1})([LSRD])([\)]{0,1})[ -]'
        x = strip_stoich_wrapper(x)
        x = re.sub(REPLACE_GROUP, lambda m: f"({SWAP_MAP[m.group(2)]})-", x)
        return x

    def remove_DL_RS(x):
        """If DL or RS, remove the info"""
        REPLACE_GROUP = r'[LSRD]{2}[ -]'
        x = strip_stoich_wrapper(x)
        x = re.sub(r'[LSRD]{2}[ -]', '', x)
        return x

    def remove_stereo(x):
        """ Remove stereoinformation """
        REPLACE_GROUP = r'[\( ][LSRD][\) ][ -]'
        x = strip_stoich_wrapper(x)
        x = re.sub(REPLACE_GROUP, "", x)
        return x

    def respell_racemic(x):
        """ remove racemic """
        x = strip_stoich_wrapper(x)
        x = re.sub(r'racemic ', '', x)
        return x

    def fix_ending(x):
        """ if compound ends with -A, switch to A instead, 
        e.g.  'butanolide-A' to 'butanolide A """
        x = strip_stoich_wrapper(x)
        x = re.sub(r'(?<=[a-zA-Z])\-(?=[a-zA-Z]$)', ' ', x)
        return x

    def respell_protocatch(x):
        """ Respell """
        x = strip_stoich_wrapper(x)
        x = re.sub(r'protocatchuic', 'protocatechuic', x)
        return x

    def fix_transport_rxns(x):
        """ Remove [side 1] or other number from transport rxns"""
        TRANSPORT_REGEX = r"[\[\(]side *\d+[\]\)]"
        x = strip_stoich_wrapper(x)
        x = re.sub(TRANSPORT_REGEX, "", x, re.IGNORECASE)
        return x

    subst_fns = [
        identity, strip_stoich_wrapper, comma_for_space, remove_plus_minus,
        swap_LS_DR, remove_DL_RS, remove_stereo, respell_racemic, fix_ending,
        respell_protocatch, fix_transport_rxns
    ]

    return subst_fns


def get_cirpy_cache(cirpy_log: Optional[str]) -> dict:
    """get_cirpy_cache.

    Args:
        cirpy_log (Optional[str]): cirpy_log file that has two cols separated
            by a tab

    Returns:
        dict:
    """

    cache = {}
    if cirpy_log and os.path.exists(cirpy_log):
        lines = open(cirpy_log, "r").readlines()
        for line in lines:
            if line.strip():
                split_line = line.strip().split("\t")
                if len(split_line) > 1:
                    cache[split_line[0].strip()] = split_line[1].strip()
                else:
                    cache[split_line[0].strip()] = None
    return cache


def cirpy_cached(mols: List[str], cirpy_log: Optional[str]) -> dict:
    """cirpy_cached.

    A way to call cirpy and stache all the outputs, whether positive or not and
    use previously outputted results. As the most expensive part of the parse,
    this saves downstream computation. 

    Args:
        mols (List[str]): mols
        cirpy_log (Optional[str]): cirpy_log

    Returns:
        dict:
    """

    cache = get_cirpy_cache(cirpy_log)
    query_mols = set(mols)
    mapping = {}

    fp = open(cirpy_log, "a") if cirpy_log else None

    for comp in query_mols:
        # If we already searched
        if comp in cache:
            logging.debug(f"Found mol {comp} in cirpy cache: {mapping[comp]}")
            if cache[comp]:
                mapping[comp] = cache[comp]
            else:
                continue
        # Try to find it with cirpy
        else:
            try:
                smiles = cirpy.resolve(comp, "smiles")
                if smiles:
                    mapping[comp] = smiles

                output_entry = f"{comp}\t{smiles}\n" if smiles else f"{comp}\n"
                # Add it to the cirpy log!
                if fp:
                    fp.write(output_entry)
            except Exception as e:
                logging.warning(f'Trouble with cirpy call for {comp}: {str(e)}')
    fp.close()
    return mapping


def query_opsin(mols: List[str],
                opsin_loc: Optional[str] = "opsin.jar") -> dict:
    """query_opsin.

    Download the .jar flie for opsin and query it 

    Args:
        mols (List[str]): mols
        opsin_loc (Optional[str]): opsin_loc

    Returns:
        dict: mapping for all recovered outputs
    """

    # If no opsin, download it
    if not os.path.exists(opsin_loc):
        utils.download_ftp(OPSIN_URL, opsin_loc)

    # Output mols to list
    with open(TEMP_OPSIN_INPUT, "w") as fp:
        fp.write("\n".join(mols))

    # Build command
    cmd = [
        "java", "-jar", opsin_loc, "-osmi", TEMP_OPSIN_INPUT, TEMP_OPSIN_OUTPUT
    ]
    # Run opsin
    try:
        cmd_out = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,  #subprocess.PIPE, 
            stderr=subprocess.DEVNULL)  #subprocess.PIPE
        # logging.debug(f"Opsin stderr: {cmd_out.stderr}")
        # logging.debug(f"Opsin stdout: {cmd_out.stdout}")
    except:
        logging.warning(f"Was unable to execute opsin command {' '.join(cmd)}")
        return {}

    # Parse results
    mapping = {}
    if os.path.exists(TEMP_OPSIN_OUTPUT):
        with open(TEMP_OPSIN_OUTPUT, "r") as fp:
            total_lines = fp.readlines()
            if len(total_lines) != len(mols):
                raise AssertionError(
                    f"{len(total_lines)} output by Opsin, expected {len(mols)}")
            for enum, line in enumerate(total_lines):
                mol = mols[enum]
                smiles = line.strip()
                # If smiles is not ""
                if smiles:
                    mapping[mol] = smiles

    # Deeete temp opsin files
    if os.path.exists(TEMP_OPSIN_OUTPUT): os.remove(TEMP_OPSIN_OUTPUT)
    if os.path.exists(TEMP_OPSIN_INPUT): os.remove(TEMP_OPSIN_INPUT)

    return mapping


###### CHEBI AND INCHI TO SMILES


def update_mapping(return_mapping: dict, temp_mapping: dict,
                   set_remaining: set) -> None:
    """update_mapping.

    Update return mapping with items in temp mapping; del from remaining

    Args:
        return_mapping (dict): return_mapping
        temp_mapping (dict): temp_mapping
        set_remaining (set): set_remaining

    Returns:
        None:
    """
    # Add all items from chebi back and REMOVE from true set
    # Make sure we add a list in update.
    for k in temp_mapping:
        item = temp_mapping[k]
        if not isinstance(item, list) or isinstance(item, set):
            item = [item]
        return_mapping[k].update(item)
        set_remaining.remove(k)


def chebi_to_smiles(chebi_set: list,
                    save_file_prefix: str,
                    use_cirpy: bool = True,
                    cirpy_log: Optional[str] = None) -> Tuple[dict, set]:
    """chebi_to_smiles.

    Args:
        chebi_set (list): chebi_set
        save_file_prefix (str): save_file_prefix
        use_cirpy (bool): If true, use cirpy
        cirpy_log (Optiona[str]): Loc of cirpy log 

    Returns:
        dict: Mapping to arrays of smiles
        set: unidentified chebi strings 
    """

    set_remaining = set(chebi_set)
    logging.info(f"Trying to map {len(set_remaining)} chebi names to smiles")
    return_mapping = defaultdict(lambda: set())
    mapping_chebi = utils.query_pubchem(
        set_remaining,
        query_type="chebi",
        save_file=f"{save_file_prefix}_pubchem_temp_chebi.txt")
    update_mapping(return_mapping, mapping_chebi, set_remaining)
    logging.info(f"After pulling from chebi, {len(set_remaining)} remaining")

    mapping_syn = utils.query_pubchem(
        set_remaining,
        query_type="synonym",
        save_file=f"{save_file_prefix}_chebi_pubchem_temp_syn.txt")

    update_mapping(return_mapping, mapping_syn, set_remaining)
    logging.info(f"After pulling from synonyms, {len(set_remaining)} remaining")

    if use_cirpy:
        set_list = list(set_remaining)
        cirpy_mapping = cirpy_cached(set_list, cirpy_log)
        update_mapping(return_mapping, cirpy_mapping, set_remaining)
        logging.info(
            f"After pulling from cirpy, {len(set_remaining)} remaining")

    return return_mapping, set_remaining


def inchi_to_smiles(inchi_set: list,
                    save_file_prefix: str,
                    use_cirpy: bool = True,
                    cirpy_log: Optional[str] = None) -> Tuple[dict, set]:
    """inchi_to_smiles.

    Args:
        inchi_set (list): inchi_set
        save_file_prefix (str): save_file_prefix
        use_cirpy (bool): If true, query cirpy
        cirpy_log (Optiona[str]): Loc of cirpy log 

    Returns:
        dict: Mapping to arrays of smiles
        set: unidentified inchi strings 
    """
    logging.info("Mapping INCHI to SMILES")
    set_remaining = set(inchi_set)
    return_mapping = defaultdict(lambda: set())
    logging.info(f"Trying to map {len(set_remaining)} inchi names to smiles")

    # First try to map these using rdkit
    for inchi in inchi_set:
        mol = Chem.MolFromInchi(inchi)
        smiles = None
        if mol:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles:
            return_mapping[inchi].add(smiles)
            set_remaining.remove(inchi)

    logging.info(
        f"After pulling from rdkit inchi, {len(set_remaining)} remaining")
    mapping_inchi = utils.query_pubchem(
        set_remaining,
        query_type="inchi",
        save_file=f"{save_file_prefix}_inchi_pubchem_temp.txt")
    update_mapping(return_mapping, mapping_inchi, set_remaining)
    logging.info(
        f"After pulling from pubchem inchi, {len(set_remaining)} remaining")

    if use_cirpy:
        # Now try to resolve with cirpy
        set_list = list(set_remaining)
        cirpy_mapping = cirpy_cached(set_list, cirpy_log)
        update_mapping(return_mapping, cirpy_mapping, set_remaining)
        logging.info(
            f"After pulling from cirpy, {len(set_remaining)} remaining")

    return return_mapping, set_remaining


# Full map to smiles procedure

def get_smiles_from_mappings(mol: str, ligand_mapping: dict,
                             inchi_mapping: dict,
                             chebi_mapping: dict) -> Optional[str]:
    """get_smiles_from_mappings.

    Return smiles string by mapping the molecule into the ligand mapping
    If ligand mapping has chebi and chebi has a smiles, return that smiles
    Else if ligand mapping has inchi and inchi has a smiles, return that smiles
    Else None

    Args:
        mol (str): mol
        ligand_mapping (dict): ligand_mapping
        inchi_mapping (dict): inchi_mapping
        chebi_mapping (dict): chebi_mapping

    Returns:
        Optional[str]: Smiles
    """
    smiles_options = None
    smiles = None

    if mol.lower() in ligand_mapping:
        ligand_entry = ligand_mapping.get(mol.lower())
        # try chebi
        if (ligand_entry["chebi"] and ligand_entry["chebi"] in chebi_mapping):
            smiles_options = chebi_mapping[ligand_entry["chebi"]]
        # try
        elif (ligand_entry["inchi"] and ligand_entry["inchi"] in inchi_mapping):
            smiles_options = inchi_mapping[ligand_entry["inchi"]]
        else:
            pass
        if smiles_options and len(smiles_options) > 1:
            logging.warning(
                f"Multiple smiles entries for {mol}, taking one at random")
        smiles = list(smiles_options)[0] if smiles_options else None

    return smiles

def resolve_from_mappings(mol_list, ligand_mapping, inchi_mapping,
                          chebi_mapping): 
    """ Wrapper around get_smiles_from_mapping that accepts a list and returns
    a dict"""
    new_mapping = {}
    for mol in mol_list: 
        smiles = get_smiles_from_mappings(mol, ligand_mapping,
                                          inchi_mapping, chebi_mapping)
        if smiles:
            new_mapping[mol] = smiles
    return new_mapping


def resolve_compound_list(mols: list,
                          ligand_mapping: dict,
                          inchi_mapping: dict,
                          chebi_mapping: dict,
                          save_prefix: str,
                          use_cirpy: bool = True,
                          cirpy_log: Optional[str] = None,
                          opsin_loc: Optional[str] = None) -> Tuple[dict, list]:
    """resolve_compound_list.


    Attempt all substitution rules on the molecule list and try to pull them
    from the inchi mapping, chebi mapping, 

    Order:
    - Try resolving with dictionaries passed in as input
    - Try resolving with pubchem
    - Try resolving with Opsin
    - Try resolving with cirpy (if use cirpy)
    - Repeat this procedure for _all_ substitution rules 

    Args:
        mols (list): mols
        ligand_mapping (dict): ligand_mapping
        inchi_mapping (dict): inchi_mapping
        chebi_mapping (dict): chebi_mapping
        save_prefix (str): save_prefix
        use_cirpy (bool): If true, make cirpy queries
        cirpy_log (Optiona[str]): Loc of cirpy log 
        opsin_loc (Optional[str]): If this is not None, download or use opsin
            from this location; requires java installed

    Returns:
        Tuple[dict, list]: mapping and unmapped compounds
    """

    mapping = dict()
    mols_remaining = set(mols)

    subst_rules = get_substitution_rules()

    # Try a number of different resolvers
    resolver_fns = []
    resolver_kwargs = []

    # Add previous dictionaries
    # Resolvers should accept a list of molecules and return a map of molecules
    # in the list to a smiles string
    resolver_fns.append(resolve_from_mappings)
    resolver_kwargs.append({"ligand_mapping": ligand_mapping, 
                            "inchi_mapping" : inchi_mapping, 
                            "chebi_mapping" : chebi_mapping})

    # convert pubchem into a mapper
    resolver_fns.append(utils.query_pubchem) 
    resolver_kwargs.append({"query_type" : "synonym",
                            "save_file" : f"{save_prefix}_pubchem_query_comp_names.txt", 
                            "return_single" : True})

    if opsin_loc:
        resolver_fns.append(query_opsin)
        resolver_kwargs.append({"opsin_loc": opsin_loc})

    if use_cirpy:
        resolver_fns.append(cirpy_cached)
        resolver_kwargs.append({"cirpy_log": cirpy_log})

    for resolver, kwargs in zip(resolver_fns, resolver_kwargs):
        resolver_name = resolver.__name__
        logging.info(f"Trying resolver: {resolver_name}")
        tested_variants = set()
        num_resolver_found = 0
        for subst_rule in subst_rules:
            subst_name = subst_rule.__name__
            mol_list = list(mols_remaining)
            logging.info(
                f"""\nRemaining molecules: {len(mols_remaining)}.\nTrying
                substitution rule \'{subst_name}\'""")
            mols_to_test = set()

            # Map the variant to its corresponding true molecule
            var_to_mol = defaultdict(lambda :  [])

            # Get all molecules with this subt rule
            for mol in mol_list:
                mol_var = subst_rule(mol)
                var_to_mol[mol_var].append(mol)
                if mol_var not in tested_variants:
                    mols_to_test.add(mol_var)
                    tested_variants.add(mol_var)

            logging.info(
                f"Mols being tried for {resolver_name} w/ this rule: {len(mols_to_test)}"
            )

            # Test this set with the resolver at hand
            resolver_mapping = resolver(list(mols_to_test), **kwargs)

            # Remap resolver mapping
            new_mapping = {true_mol : smiles for k,smiles in resolver_mapping.items() for true_mol in var_to_mol[k]}

            mapping.update(new_mapping)
            mols_remaining = mols_remaining.difference(
                set(new_mapping.keys()))
            num_resolver_found += len(new_mapping)
            logging.info(
                f"""After pulling from {resolver_name}, {len(mols_remaining)} remaining.
                {len(resolver_mapping)} found with {subst_name} + {resolver_name}.
                {num_resolver_found} found with {resolver_name} so far""")

    return mapping, list(mols_remaining)

########## Standardization ##########

def standardize_mol(standardizer: object, smiles_string: str, 
                    taut_cutoff: int = 50) -> str:
    """standardize_mol.

    Args:
        standardizer (object): standardizer
        smiles_string (str): smiles_string
        taut_cutoff (int): standardize mols for common tautomers if they have
            <= this length

    Returns:
        str:
    """
    if not standardizer:
        standardizer = molvs.Standardizer()

    mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
    # No stereotype 
    mol = standardizer.standardize(mol)

    # Don't use charge parent to avoid fragmenting
    mol = standardizer.uncharge(mol) 

    # mol = standardizer.isotope_parent(mol, skip_standardize = True) 
    # Preserve stereochemistry?
    # mol = standardizer.stereo_parent(mol, skip_standardize = True) 
    if mol.GetNumAtoms() <= taut_cutoff: 
        mol = standardizer.tautomer_parent(mol, skip_standardize = True)

    mol = standardizer.standardize(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def standardize_chunk(smiles_strs : List[str], 
                      current_cache : dict,
                      standardizer : object) -> Tuple[dict, list]: 
    """standardize_chunk.

    Args:
        smiles_strs (List[str]): smiles_strs
        current_cache (dict): current_cache
        standardizer (dict): object

    Returns:
        dict, list: mappings, num failed
    """
    failed_list = []
    new_dict = {}
    for index, v in enumerate(smiles_strs):
        try:
            # Try standardizing
            if v in current_cache: 
                standardized = current_cache[v]
            else:
                standardized = standardize_mol( standardizer, v)

            # If we get a None smiles, jump out to the except clause
            if not standardized:
                raise ValueError

            new_dict[v] = standardized
        except:
            failed_list.append(v)

    return new_dict, failed_list


def standardize_mols(mapped_compounds : dict, standardizer_log : str = None, 
                     save_freq : int = 1000, multiprocess_count : int = 1) -> dict: 
    """standardize_mols.

    Args:
        mapped_compounds (dict): mapped_compounds
        standardizer_log (str):  
        save_freq (int) : how often to repickle
        multiprocess_count (int) : How many multiprocessses to run

    Returns:
        dict:
    """

    def unique(temp_list):
        """Order preserving unique fn for testing"""
        seen = set()
        return [x for x in temp_list if not (x in seen or seen.add(x))]

    # First get standardizer
    # Use all default transformations
    # This runs faster
    standardizer = molvs.Standardizer()

    logging.info("Starting to standardize molecules")

    # Get unique smiles we need to identify 
    to_standardize = unique(mapped_compounds.values())

    if standardizer_log and os.path.isfile(standardizer_log): 
        logging.info("Loading previous standardization mappings")
        standardized_mapped = utils.load_json(standardizer_log)
        logging.info(f"Loaded {len(standardized_mapped)} previous standardizations")
    else: 
        logging.info("Unable to load previous mappings")
        standardized_mapped = {}

    # If multiprocess
    pool = None
    if multiprocess_count > 1:
        pool = multiprocessing.Pool()
        

    # Mapping of unique smiles to standardized smiles
    num_failed = 0
    # Make groups of the size
    division_factor = save_freq * multiprocess_count
    num_groups = len(to_standardize) // division_factor
    # sort to_standardize..
    to_standardize = sorted(to_standardize, key=len)
    chunk_groups = np.array_split(to_standardize, num_groups) 
    for chunk_index, mol_chunk in enumerate(chunk_groups):

        if pool:  
            results = []
            # Start all sub jobs
            for sub_chunk in np.array_split(mol_chunk, multiprocess_count):
                results.append(pool.apply_async(standardize_chunk, 
                                                args=(sub_chunk, 
                                                      standardized_mapped, 
                                                      standardizer)))
            # Wait for jobs -- apparently .get() blocks, so no need
            #pool.join()
            # Aggregate
            new_mapping = {}
            failed = []
            for r in results: 
                new_dict_sub, failed_sub = r.get()
                new_mapping.update(new_dict_sub)
                failed.extend(failed_sub)
        else: 
            new_mapping, failed = standardize_chunk(mol_chunk,
                                                    standardized_mapped,
                                                    standardizer) 

        # Aggregate the results
        standardized_mapped.update(new_mapping)

        # Stache results for later just in case
        if standardizer_log: 
            utils.dump_json(standardized_mapped, standardizer_log)

        # Log the failures
        num_failed += len(failed)
        for failure in failed: 
            logging.warning(f"Failed to map {failure}")
        logging.info(f"Mapped {len(new_mapping)} new compounds in chunk {chunk_index}")

    if pool: pool.close()
    # Set equal!
    mapped_compounds = {k : standardized_mapped[v] for k,v in mapped_compounds.items() if v in standardized_mapped}
    # Redo the mapping
    logging.info("Finished standardizing molecules")
    logging.info(f"Number failed: {num_failed}")

    return mapped_compounds 
